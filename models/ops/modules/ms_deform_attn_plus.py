import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..functions import MSDeformAttnFunction
from torch.nn.init import xavier_uniform_, constant_

def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    return (n & (n-1) == 0) and n != 0

class MSDeformAttn(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4, sigmoid_attn=False, attn_mode='spatial_attn'):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
        _d_per_head = d_model // n_heads
        # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_head):
            warnings.warn("You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2 "
                          "which is more efficient in our CUDA implementation.")

        self.im2col_step = 64        
        self.sigmoid_attn = sigmoid_attn

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.attn_mode = attn_mode

        if attn_mode == 'spatial_attn':
            self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
            
        elif attn_mode == 'temporal_attn':
            self.head_proj = nn.Linear(d_model, d_model)
            #self.motion_offset = nn.Linear(d_model, n_heads * n_levels  * 2)

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        # viz
        self.viz = None
        self._reset_parameters()
        
    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        if self.attn_mode == 'spatial_attn':
            constant_(self.attention_weights.weight.data, 0.)
            constant_(self.attention_weights.bias.data, 0.)
        elif self.attn_mode == 'temporal_attn':
            xavier_uniform_(self.head_proj.weight.data)
            constant_(self.head_proj.bias.data, 0.)

        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, 
                      input_level_start_index, input_padding_mask=None, track_pos_embed=None):
        '''
        query: (N, Length_{query}, C)
            encoder:
                feature queries: (N, sum(Hi*Wi), C)
            decoder:
                track_queries: (N, Length_{tacklets}, C)
                detect_queries: (N, Length_{detection}, C)
        '''
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)
        sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)

        #if self.attn_mode == 'temporal_attn':
            #motion_pred_offset = self.motion_offset(track_pos_embed).view(N, Len_q, self.n_heads, self.n_levels, 1, 2)
            #sampling_offsets = torch.cat([sampling_offsets, motion_pred_offset], dim=-2)
        # N, Len_q, n_heads, n_levels, n_points, 2
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                                 + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                                 + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))

        if self.attn_mode == 'spatial_attn':
            attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)
            attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)
            output = MSDeformAttnFunction.apply(
                    value, input_spatial_shapes, input_level_start_index, sampling_locations, attention_weights, self.im2col_step)

        elif self.attn_mode == 'temporal_attn':
            if query.size(1) == 0:
                output = query
            else:
                pre_state = self.head_proj(query).view(N, Len_q, self.n_heads, self.d_model//self.n_heads)
                output, attn_weights = deform_temporal_attn_core_pytorch(
                    value, pre_state, input_spatial_shapes, sampling_locations, sigmoid_attn=self.sigmoid_attn)
                
                self.viz = (reference_points.clone(), sampling_locations.clone(), attn_weights.clone())

        output = self.output_proj(output)
        return output

def deform_temporal_attn_core_pytorch(value, pre_state, value_spatial_shapes, sampling_locations, sigmoid_attn=False):
    '''
    value: current frame feature
    pre_value: previous frame feature
    temporal offset: object queries of last frame make a guess of the temporal offset w.r.t the current frame
    '''

    N, seq_len, n_heads, head_dim = value.shape
    N, num_tracks, n_heads, n_levels, n_points, _ = sampling_locations.shape
    N, num_tracks, n_heads, head_dim = pre_state.shape

    # N, num_tracks, nhead, head_dim
    pre_state_ = pre_state.view(N, num_tracks*n_heads, head_dim, 1).transpose(2, 3)
    value = value.split([H*W for H, W in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1 # [-1, 1]
    sampling_value_list = []

    for level_id, (H, W) in enumerate(value_spatial_shapes):
        # N, H*W, n_head, head_dim -> N, H*W, d_model -> N, d_model, H*W -> N*n_head, head_dim, H, W
        value_layer = value[level_id].flatten(2).transpose(1, 2).reshape(N*n_heads, head_dim, H, W)

        # by looking into frame t-1, we can have some prior to where current frame instance may occur(i.e:backward flow)
        # N, offset_len, n_head, n_level, n_point, 2 -> N, n_head, offset_len, n_level, n_point, 2 -> N*n_head, offset_len, n_level, n_point, 2
        sampling_grids_layer = sampling_grids[:, :, :, level_id].transpose(1, 2).flatten(0, 1)

        # get instance value from backward flow
        # N*n_head, head_dim, seq_len, n_point
        sampling_value_layer = F.grid_sample(value_layer, sampling_grids_layer, mode='bilinear', 
                                             padding_mode='zeros', align_corners=False)
        sampling_value_list.append(sampling_value_layer)
    
    # N*n_head, head_dim, seq_len, n_level, n_point -> N, num_tracks*n_head, head_dim, n_level*n_point
    sampling_value_flatten = torch.stack(sampling_value_list, dim=-2) \
                            .view(N, n_heads, head_dim, num_tracks, n_levels*n_points) \
                            .permute(0, 3, 1, 2, 4) \
                            .flatten(1, 2)
    # N, num_tracks*n_head, 1, n_level*n_point 
    scores = torch.matmul(pre_state_, sampling_value_flatten) / math.sqrt(head_dim)

    if sigmoid_attn:
        attn_weights = scores.sigmoid()
    else:
        attn_weights = F.softmax(scores, dim=-1)

    output = torch.matmul(attn_weights, sampling_value_flatten.transpose(-2, -1)).unsqueeze(-2)
    return output.view(N, num_tracks, n_heads * head_dim), attn_weights