"""
Module containing the decoders.
"""
import numpy as np

import torch
from torch import nn


# ALL decoders should be called Decoder<Model>
def get_decoder(model_type):
    #model_type = model_type.lower().capitalize()
    return eval("Decoder{}".format(model_type))


# class DecoderBurgess(nn.Module):
#     def __init__(self, img_size,
#                  latent_dim=128, num_queries=10):
#         r"""Decoder of the model proposed in [1].

#         Parameters
#         ----------
#         img_size : tuple of ints
#             Size of images. E.g. (1, 32, 32) or (3, 64, 64).

#         latent_dim : int
#             Dimensionality of latent output.

#         Model Architecture (transposed for decoder)
#         ------------
#         - 4 convolutional layers (each with 32 channels), (4 x 4 kernel), (stride of 2)
#         - 2 fully connected layers (each of 256 units)
#         - Latent distribution:
#             - 1 fully connected layer of 20 units (log variance and mean for 10 Gaussians)

#         References:
#             [1] Burgess, Christopher P., et al. "Understanding disentangling in
#             $\beta$-VAE." arXiv preprint arXiv:1804.03599 (2018).
#         """
#         super(DecoderBurgess, self).__init__()

#         # Layer parameters
#         hid_channels = 64
#         kernel_size = 4
#         hidden_dim = 256
#         self.img_size = img_size
#         self.num_queries = num_queries
#         # Shape required to start transpose convs
#         self.reshape = (hid_channels, kernel_size, kernel_size)
#         n_chan = self.img_size[0]
#         self.img_size = img_size

#         # Fully connected layers
#         self.lin1 = nn.Linear(latent_dim, hidden_dim)
#         self.lin2 = nn.Linear(hidden_dim, hidden_dim)
#         self.lin3 = nn.Linear(hidden_dim, np.product(self.reshape))

#         # Convolutional layers
#         cnn_kwargs = dict(stride=2, padding=1)
#         # If input image is 64x64 do fourth convolution
#         if self.img_size[1] >= 64: #== self.img_size[2] == 64:
#             self.convT_64 = nn.ConvTranspose2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)

#         self.convT1 = nn.ConvTranspose2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)
#         self.convT2 = nn.ConvTranspose2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)
#         self.convT3 = nn.ConvTranspose2d(hid_channels, n_chan, kernel_size, **cnn_kwargs)

#     def forward(self, z):
#         batch_size = z.size(0)

#         # Fully connected layers with ReLu activations
#         x = torch.relu(self.lin1(z))
#         x = torch.relu(self.lin2(x))
#         x = torch.relu(self.lin3(x))
#         x = x.view(batch_size*self.num_queries, *self.reshape)

#         # Convolutional layers with ReLu activations
#         if self.img_size[1] >= 64: #self.img_size[1] == self.img_size[2] == 64:
#             x = torch.relu(self.convT_64(x))
#         if self.img_size[1] >= 128:  #self.img_size[1] == self.img_size[2] == 128:
#             x = torch.relu(self.convT_64(x))
            
#         x = torch.relu(self.convT1(x))
#         x = torch.relu(self.convT2(x))
#         # Sigmoid activation for final conv layer
#         x = torch.sigmoid(self.convT3(x))

#         x = x.view(batch_size, self.num_queries, *self.img_size)
#         return x
    
    
class DecoderBurgess(nn.Module):
    def __init__(self, img_size,
                 latent_dim=128, num_queries=10):
        r"""Decoder of the model proposed in [1].

        Parameters
        ----------
        img_size : tuple of ints
            Size of images. E.g. (1, 32, 32) or (3, 64, 64).

        latent_dim : int
            Dimensionality of latent output.

        Model Architecture (transposed for decoder)
        ------------
        - 4 convolutional layers (each with 32 channels), (4 x 4 kernel), (stride of 2)
        - 2 fully connected layers (each of 256 units)
        - Latent distribution:
            - 1 fully connected layer of 20 units (log variance and mean for 10 Gaussians)

        References:
            [1] Burgess, Christopher P., et al. "Understanding disentangling in
            $\beta$-VAE." arXiv preprint arXiv:1804.03599 (2018).
        """
        super(DecoderBurgess, self).__init__()

        # Layer parameters
        hid_channels = 16
        kernel_size = 8
        hidden_dim = 128
        self.img_size = img_size
        self.num_queries = num_queries
        # Shape required to start transpose convs
        self.reshape = (hid_channels, 2*kernel_size, 2*kernel_size)
        n_chan = self.img_size[0]
        self.img_size = img_size

        # Fully connected layers
        self.lin1 = nn.Linear(latent_dim, 4*hidden_dim)
        self.lin2 = nn.Linear(4*hidden_dim, 8*hidden_dim)
        self.lin3 = nn.Linear(8*hidden_dim, 16*hidden_dim)
        self.lin4 = nn.Linear(16*hidden_dim, np.product(self.reshape))

        kernel_size = 4
        # Convolutional layers
        cnn_kwargs = dict(stride=2, padding=1)
        # If input image is 64x64 do fourth convolution
#         if self.img_size[1] >= 64: #== self.img_size[2] == 64:
#             self.convT_64 = nn.ConvTranspose2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)

        self.convT1 = nn.ConvTranspose2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)
        self.convT2 = nn.ConvTranspose2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)
        self.convT3 = nn.ConvTranspose2d(hid_channels, n_chan, kernel_size, **cnn_kwargs)

    def forward(self, z):
        batch_size = z.size(0)

        # Fully connected layers with ReLu activations
        x = torch.relu(self.lin1(z))
        x = torch.relu(self.lin2(x))
        x = torch.relu(self.lin3(x))
        x = torch.relu(self.lin4(x))
        x = x.view(batch_size*self.num_queries, *self.reshape)

        # Convolutional layers with ReLu activations
#         if self.img_size[1] >= 64: #self.img_size[1] == self.img_size[2] == 64:
#             x = torch.relu(self.convT_64(x))
#         if self.img_size[1] >= 128:  #self.img_size[1] == self.img_size[2] == 128:
#             x = torch.relu(self.convT_64(x))
            
        #x = torch.relu(self.convT1(x))
        x = torch.relu(self.convT2(x))
        # Sigmoid activation for final conv layer
        x = torch.sigmoid(self.convT3(x))

        x = x.view(batch_size, self.num_queries, *self.img_size)
        return x
    
    
    
class DecoderOCRA(nn.Module):
    """
    OCRA model is trained to detect, classify, and reconstruct the objects in the image. 
    This model 
        1) encodes an image through recurrent read operations that forms a spatial attention on objects
        2) effectively binds features and classify objects through capsule representation and its dynamic routing
        3) decodes/reconstructs an image from class capsules through recurrent write operations 
    """
        
    def __init__(self, args, img_size, latent_dim=128, num_queries=10):
        super().__init__()
        
        self.task = args.task
        # dataset info
        self.C, self.H, self.W = args.im_dims
        self.image_dims = args.im_dims

        #self.num_classes = args.num_classes # number of categories 
      
        self.use_write_attn = True #args.use_write_attn  
            
        if self.use_write_attn: 
            self.write_size = 18 #args.write_size
        else:
            self.write_size = self.H
            
        self.time_steps = 10 # args.write_time_steps
        self.hidden_dim = args.hidden_dim
        self.lstm_size = 512 #args.lstm_size
            
        # decoder RNN/reconstruction  
        self.decoder = nn.LSTMCell(self.hidden_dim, self.lstm_size)   
        self.recon_model = True # args.recon_model # whether decoder generates a reconstruction of the input         
        self.write_linear = nn.Linear(self.lstm_size, self.write_size * self.write_size *self.C) # write layer getting input from decoder RNN and generating what will be written to the canvas 
        self.relu = nn.ReLU()
        
        # attention layers that output 4 params that specify NxN array of gaussian filters      
        self.write_attention_linear = nn.Linear(self.lstm_size, 4)
        

    def write(self, h_dec):
        # get write_size * write_size writing patch (if no write attn is used, write_size is set to be image height)
        w_g = self.write_linear(h_dec)
        
        if self.use_write_attn: 
            # get attention params and compute gaussian filterbanks
            g_x, g_y, logvar, logdelta = self.write_attention_linear(h_dec).split(split_size=1, dim=1) #removed loggamma 
            g_x, g_y, delta, mu_x, mu_y, F_x, F_y = compute_filterbank_matrices(g_x, g_y, logvar, logdelta, self.H, self.W, self.write_size)
           
            # expand filterbanks to have input channel dimension 
            # [n_batch, n_channel, im_x or im_h] --> [n_batch, n_channel, write_attn_size, im_x or im_h] 
            F_y = torch.unsqueeze(F_y, 1)
            F_y = F_y.repeat(1, self.C, 1, 1) 
            F_x = torch.unsqueeze(F_x, 1)
            F_x = F_x.repeat(1, self.C, 1, 1) 
            
            # apply filterbanks and get read output in original coords H*W 
            # [B,C,H,W] = [B,C,H,N] @ [B,C,N,N] @ [B,C,N,W]            
            w = F_y.transpose(-2, -1) @ w_g.view(-1, self.C, self.write_size, self.write_size) @ F_x
            write_att_param = [g_x, g_y, delta, mu_x, mu_y, F_x, F_y, w_g.detach()]
            
            return  w.view(w.shape[0], -1), write_att_param 

        else: # if not using write attention, just use raw write output from the decoder            
            return w_g, []

    def forward(self, x, y=None):
        
        batch_size = x.shape[0]
        device = x.device
        
        # intitalize the hidden/cell state for the decoder
        h_dec = torch.zeros(batch_size, self.lstm_size).to(device)
        c_dec = torch.zeros_like(h_dec)
        
        # initialize the canvas 
        c = torch.zeros(x.shape).to(device)
        c_step = torch.zeros(batch_size, self.time_steps, self.C*self.H*self.W).to(device) 

        # run model forward
        for t in range(self.time_steps):
                      
            # feed objectcaps to the decoder RNN
            h_dec, c_dec = self.decoder(x, (h_dec, c_dec))  
            
            # whether do reconstruction from decoder outputs
            # get write canvas ouput
            c_write, write_att_param = self.write(h_dec)
            c_write = self.relu(c_write) # only positive segments are written to the canvas, prevents deleting of earlier writes 
            c_step[:,t:t+1,:] = torch.unsqueeze(c_write, 1) # keep whats written to the canvas at each step
            
        '''
        Returns
        objcaps_len_step -- length of object capsules at each timestep
        read_x_step -- Sum of all reads from image (use for masking reconstruction error more focused on the read parts)
        c_step --- list of canvases at each timestep
        
        
        old stuff
        c -- tensors of shape (B, 1); final cumulative reconstruction canvas
        y_pred -- cumulative length of object capsules; final prediction vector 
        objcaps_len_step -- length of object capsules at each timestep 
        att_param_step -- all model outputs stored at each step
            0) read attention param 
            1) read output 
            2) write attention param
            3) write output (canvas) 
            4) cumulative canvas 
            5) class masked canvas 
            6) coupling coefficients
        '''
        
        return c_step.sum(1)
    

def compute_filterbank_matrices(g_x, g_y, logvar, logdelta, H, W, attn_window_size):
    """ DRAW section 3.2 -- computes the parameters for an NxN grid of Gaussian filters over the input image.
    
    note. B = batch dim; N = attn window size; H = original heigh; W = original width
    
    Args 
        g_x, g_y -- tensors of shape (B, 1); unnormalized center coords for the attention window
        logvar -- tensor of shape (B, 1); log variance for the Gaussian filters (filterbank matrices) on the attention window
        logdelta -- tensor of shape (B, 1); unnormalized stride for the spacing of the filters in the attention window
        H, W -- scalars; original image dimensions
        attn_window_size -- scalar; size of the attention window (specified by the read_size / write_size input args

    Returns
        g_x, g_y -- tensors of shape (B, 1); normalized center coords of the attention window;
        delta -- tensor of shape (B, 1); stride for the spacing of the filters in the attention window
        mu_x, mu_y -- tensors of shape (B, attn_window_size); means location of the filters at row and column
        F_x, F_y -- tensors of shape (B, N, W) and (B, N, H) where N=attention_window_size; filterbank matrices
    """
    batch_size = g_x.shape[0]
    device = g_x.device

    # rescale attention window center coords and stride to ensure the initial patch covers the whole input image
    # eq 22 - 24
    g_x = 0.5 * (W + 1) * (g_x + 1)  # (B, 1)
    g_y = 0.5 * (H + 1) * (g_y + 1)  # (B, 1)
    delta = (max(H, W) - 1) / (attn_window_size - 1) * logdelta.exp()  # (B, 1)

    # compute the means of the filter
    # eq 19 - 20
    mu_x = g_x + (torch.arange(1., 1. + attn_window_size).to(device) - 0.5*(attn_window_size + 1)) * delta  # (B, N)
    mu_y = g_y + (torch.arange(1., 1. + attn_window_size).to(device) - 0.5*(attn_window_size + 1)) * delta  # (B, N)

    # compute the filterbank matrices
    # eq 25 -- combines logvar=(B, 1, 1) * ( range=(B, 1, W) - mu=(B, N, 1) ) = out (B, N, W); then normalizes over W dimension;
    F_x = torch.exp(- 0.5 / logvar.exp().view(-1,1,1) * (torch.arange(1., 1. + W).repeat(batch_size, 1, 1).to(device) - mu_x.unsqueeze(-1))**2)
    F_x = F_x / torch.sum(F_x + 1e-8, dim=2, keepdim=True)  # normalize over the coordinates of the input image
    # eq 26
    F_y = torch.exp(- 0.5 / logvar.exp().view(-1,1,1) * (torch.arange(1., 1. + H).repeat(batch_size, 1, 1).to(device) - mu_y.unsqueeze(-1))**2)
    F_y = F_y / torch.sum(F_y + 1e-8, dim=2, keepdim=True)  # normalize over the coordinates of the input image

    # return rescaled attention window center coords and stride + gaussian filters 
    return g_x, g_y, delta, mu_x, mu_y, F_x, F_y
