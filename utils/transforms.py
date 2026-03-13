import numpy as np
import torch

class Transform3DForM3DCLIP:
    def __init__(self, target_depth=32, target_height=256, target_width=256):
        self.target_depth = target_depth
        self.target_height = target_height
        self.target_width = target_width
        
        
    def tile_slice_to_32(self,volume_tensor: torch.Tensor) -> torch.Tensor:
        """
        Given a tensor of shape (B, C, D, H, W) where D=1, 
        replicate the single slice to create (B, C, 32, H, W) for M3D-CLIP.
        """
        #print(f"Input to tile_slice_to_32: {volume_tensor.shape}")
        
        B, C, D, H, W = volume_tensor.shape
        
        if D != 1:
            raise ValueError(f"Expected depth=1, got depth={D}")
        
        # Remove the single depth dimension and add it back with 32 repetitions
        single_slice = volume_tensor.squeeze(2)  # [B, C, H, W]
        single_slice = single_slice.unsqueeze(2)  # [B, C, 1, H, W]
        
        # Repeat along depth dimension to get 32 slices
        tiled_volume = single_slice.repeat(1, 1, 32, 1, 1)  # [B, C, 32, H, W]
        
        #print(f"Output from tile_slice_to_32: {tiled_volume.shape}")
        return tiled_volume

    def __call__(self, volume_np):
        """
        Args:
            volume_np (np.ndarray): Input 3D volume, assumed to be HxWxD or DxHxW.
                                     This function will try to detect and handle it.
        Returns:
            torch.Tensor: Transformed volume of shape (1, target_depth, target_height, target_width)
                          and normalized to 0-1.
        """
        # Ensure it's a NumPy array
        if not isinstance(volume_np, np.ndarray):
            raise TypeError("Input must be a NumPy array.")

        # 0. Convert to float32 for calculations and PyTorch
        volume_np = volume_np.astype(np.float32)

        
                # handle full‐volume HxWxD → DxHxW
        if volume_np.shape == (240, 240, 155):
            volume_np = np.transpose(volume_np, (2, 0, 1))  # → (155,240,240)
        # if already depth‐first shapes, we accept ANY spatial size:
        elif volume_np.shape[0] in (155, 32, 1):
            # volume_np is already (D, H_crop, W_crop)
            pass
        else:
            raise ValueError(
                f"Unexpected input volume shape: {volume_np.shape}. "
                "Expected HxWxD=(240,240,155) or DxHxW with depth 1,32,155."
            )

        # 2. Convert NumPy array to PyTorch Tensor
        volume_tensor = torch.from_numpy(volume_np)

        # 3. Add Channel Dimension (C=1)
        # Reshape to (C, D, H, W) for PyTorch's interpolate
        if volume_tensor.ndim == 3: # DxHxW
            volume_tensor = volume_tensor.unsqueeze(0) # Now CxDxHxW: (1, 155, 240, 240)
        elif volume_tensor.ndim == 4 and volume_tensor.shape[0] == 1: # Already CxDxHxW
            pass
        else:
            raise ValueError(f"Tensor shape after channel add is unexpected: {volume_tensor.shape}")


        # 4. Resize/Resample to target_depth, target_height, target_width
        # M3D-CLIP wants D=32, H=256, W=256
        # torch.nn.functional.interpolate expects input (N, C, D, H, W) or (N, C, H, W) or (N, C, W)
        # Our tensor is (C, D, H, W), so add batch dim N=1 temporarily
        volume_tensor_batched = volume_tensor.unsqueeze(0) # Now (1, 1, 155, 240, 240)
        
        
        if volume_tensor_batched.shape[2] == 1:
            # If depth is 1, tile the slice to create 32 slices
            volume_tensor_batched = self.tile_slice_to_32(volume_tensor_batched)

        resized_volume = torch.nn.functional.interpolate(
            volume_tensor_batched,
            size=(self.target_depth, self.target_height, self.target_width),
            mode='trilinear',  # For 3D image data
            align_corners=False # Generally recommended for 'trilinear'
        )
        # Remove the temporary batch dimension
        resized_volume = resized_volume.squeeze(0) # Back to (1, 32, 256, 256)

        # 5. Min-Max Normalization to 0-1 range (per volume)
        min_val = resized_volume.min()
        max_val = resized_volume.max()
        if max_val - min_val > 1e-6: # Avoid division by zero if volume is constant
            normalized_volume = (resized_volume - min_val) / (max_val - min_val)
        else:
            normalized_volume = torch.zeros_like(resized_volume) # Or handle as appropriate

        return normalized_volume

class Transform3DMask:
    def __init__(self, target_height=256, target_width=256,target_depth=32):
        # Depth is fixed by sub-volume chunking (32 slices)
        self.target_height = target_height
        self.target_width = target_width
        self.target_depth = target_depth

    def __call__(self, sub_mask_np_d32hw): # Expects (32, H_orig, W_orig)
        """
        Transforms a 3D sub-mask to match the expected shape and data type of M3D-CLIP.

        Args:
            sub_mask_np_d32hw (np.ndarray): Input 3D sub-mask of shape (32, H_orig, W_orig).

        Returns:
            torch.Tensor: Transformed sub-mask of shape (1, 32, target_height, target_width) and data type torch.uint8.
        """

        # Convert to float32 for interpolation (works better with float)
        mask_np = sub_mask_np_d32hw.astype(np.float32)

        # Convert NumPy array to PyTorch Tensor
        mask_tensor = torch.from_numpy(mask_np) # Shape: (32, H_orig, W_orig)

        # Add channel dimension (C=1) if not already present
        if mask_tensor.ndim == 3:
            mask_tensor = mask_tensor.unsqueeze(0) # Now (1, 32, H_orig, W_orig)

        # Add batch dimension (N=1) temporarily for interpolation
        mask_tensor_batched = mask_tensor.unsqueeze(0) # Now (1, 1, 32, H_orig, W_orig)

        # Resize/Resample to target shape using nearest neighbor interpolation
        resized_mask = torch.nn.functional.interpolate(
            mask_tensor_batched,
            size=(self.target_depth, self.target_height, self.target_width), # (D, H, W)
            mode='nearest' # Crucial for masks
        )

        # Remove the temporary batch dimension
        resized_mask = resized_mask.squeeze(0) # Back to (1, 32, target_height, target_width)

        # Convert to uint8 (or long) for compatibility with M3D-CLIP
        return resized_mask.byte() 
    