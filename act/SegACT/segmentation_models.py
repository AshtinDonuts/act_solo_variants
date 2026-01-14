"""
Segmentation model loaders for SegACT.

Supports multiple segmentation models:
- FastSAM: Fast Segment Anything Model (via ultralytics)
- RE_DETR: Referring Expression DETR (for object segmentation)
- Placeholder: Simple grayscale conversion (default)
"""
import torch
import numpy as np
from typing import Optional, Union, Dict
import warnings


class SegmentationModelBase:
    """Base class for segmentation models"""
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.model = None
    
    def segment(self, image: torch.Tensor) -> torch.Tensor:
        """
        Generate segmentation mask from RGB image.
        
        Args:
            image: RGB image tensor (C, H, W) or (batch, C, H, W) or (batch, num_cam, C, H, W)
        
        Returns:
            Segmentation mask tensor with same shape as input
        """
        raise NotImplementedError


class PlaceholderSegmentation(SegmentationModelBase):
    """
    Placeholder segmentation: simple grayscale conversion.
    Used as fallback when no segmentation model is specified.
    """
    def __init__(self, device: str = 'cuda'):
        super().__init__(device)
    
    def segment(self, image: torch.Tensor) -> torch.Tensor:
        """Convert RGB to grayscale as placeholder segmentation"""
        if len(image.shape) == 5:  # (batch, num_cam, C, H, W)
            seg = torch.mean(image, dim=2, keepdim=True)  # (batch, num_cam, 1, H, W)
            seg = seg.repeat(1, 1, 3, 1, 1)  # (batch, num_cam, 3, H, W)
            return seg
        elif len(image.shape) == 4:  # (batch, C, H, W) or (num_cam, C, H, W)
            # Check if it's batch or num_cam by checking if C is 3
            if image.shape[1] == 3:  # (batch, C, H, W)
                seg = torch.mean(image, dim=1, keepdim=True)  # (batch, 1, H, W)
                seg = seg.repeat(1, 3, 1, 1)  # (batch, 3, H, W)
            else:  # (num_cam, C, H, W)
                seg = torch.mean(image, dim=1, keepdim=True)  # (num_cam, 1, H, W)
                seg = seg.repeat(1, 3, 1, 1)  # (num_cam, 3, H, W)
            return seg
        elif len(image.shape) == 3:  # (C, H, W)
            seg = torch.mean(image, dim=0, keepdim=True)  # (1, H, W)
            seg = seg.repeat(3, 1, 1)  # (3, H, W)
            return seg
        else:
            raise ValueError(f"Unexpected image shape: {image.shape}")


class FastSAMSegmentation(SegmentationModelBase):
    """
    FastSAM (Fast Segment Anything Model) via ultralytics.
    
    FastSAM is a faster alternative to SAM that can segment objects in images.
    """
    def __init__(self, model_path: Optional[str] = None, device: str = 'cuda', 
                 conf_threshold: float = 0.25, iou_threshold: float = 0.45):
        """
        Args:
            model_path: Path to FastSAM model weights (None = use default)
            device: Device to run model on
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
        """
        super().__init__(device)
        try:
            from ultralytics import SAM
            self.sam_model = SAM(model_path or 'FastSAM-x.pt')  # Will download if not present
            self.conf_threshold = conf_threshold
            self.iou_threshold = iou_threshold
        except ImportError:
            raise ImportError(
                "ultralytics not installed. Install with: pip install ultralytics"
            )
        except Exception as e:
            warnings.warn(f"Failed to load FastSAM: {e}. Falling back to placeholder.")
            self.sam_model = None
    
    def segment(self, image: torch.Tensor) -> torch.Tensor:
        """
        Generate segmentation mask using FastSAM.
        
        Args:
            image: RGB image tensor, normalized to [0, 1]
                  Shape: (C, H, W), (batch, C, H, W), or (batch, num_cam, C, H, W)
        
        Returns:
            Segmentation mask tensor with same shape as input
        """
        if self.sam_model is None:
            # Fallback to placeholder
            placeholder = PlaceholderSegmentation(self.device)
            return placeholder.segment(image)
        
        # Handle different input shapes
        original_shape = image.shape
        is_batch = len(image.shape) == 5  # (batch, num_cam, C, H, W)
        is_single_batch = len(image.shape) == 4  # (batch, C, H, W) or (num_cam, C, H, W)
        
        if is_batch:
            batch_size, num_cam, C, H, W = image.shape
            results = []
            for b in range(batch_size):
                cam_results = []
                for c in range(num_cam):
                    seg_mask = self._segment_single_image(image[b, c])
                    cam_results.append(seg_mask)
                results.append(torch.stack(cam_results, dim=0))
            return torch.stack(results, dim=0)
        elif is_single_batch:
            # Could be (batch, C, H, W) or (num_cam, C, H, W)
            if image.shape[0] > 1 and image.shape[1] == 3:
                # Likely (num_cam, C, H, W)
                results = []
                for i in range(image.shape[0]):
                    seg_mask = self._segment_single_image(image[i])
                    results.append(seg_mask)
                return torch.stack(results, dim=0)
            else:
                # Likely (batch, C, H, W)
                results = []
                for i in range(image.shape[0]):
                    seg_mask = self._segment_single_image(image[i])
                    results.append(seg_mask)
                return torch.stack(results, dim=0)
        else:
            # Single image (C, H, W)
            return self._segment_single_image(image)
    
    def _segment_single_image(self, image: torch.Tensor) -> torch.Tensor:
        """Segment a single image (C, H, W)"""
        # Convert to numpy and denormalize if needed
        if image.max() <= 1.0:
            # Assume normalized [0, 1], convert to [0, 255]
            img_np = (image.detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        else:
            img_np = image.detach().cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
        
        # Run FastSAM
        try:
            results = self.sam_model(img_np, conf=self.conf_threshold, iou=self.iou_threshold)
            
            # Get segmentation masks
            if len(results) > 0 and hasattr(results[0], 'masks'):
                masks = results[0].masks
                if masks is not None and masks.data is not None:
                    # Combine all masks into one
                    combined_mask = torch.zeros((img_np.shape[0], img_np.shape[1]), 
                                               dtype=torch.float32, device=image.device)
                    for mask in masks.data:
                        mask_tensor = mask.cpu().float()
                        # Resize mask to image size if needed
                        if mask_tensor.shape != combined_mask.shape:
                            import torch.nn.functional as F
                            mask_tensor = F.interpolate(
                                mask_tensor.unsqueeze(0).unsqueeze(0),
                                size=(img_np.shape[0], img_np.shape[1]),
                                mode='bilinear',
                                align_corners=False
                            ).squeeze()
                        combined_mask = torch.maximum(combined_mask, mask_tensor)
                    
                    # Convert to 3-channel
                    seg_mask = combined_mask.unsqueeze(0).repeat(3, 1, 1)  # (3, H, W)
                    return seg_mask.to(image.device)
            
            # No masks found, return grayscale
            seg_mask = torch.mean(image, dim=0, keepdim=True).repeat(3, 1, 1)
            return seg_mask
            
        except Exception as e:
            warnings.warn(f"FastSAM segmentation failed: {e}. Using placeholder.")
            # Fallback to grayscale
            seg_mask = torch.mean(image, dim=0, keepdim=True).repeat(3, 1, 1)
            return seg_mask


class REDETRSegmentation(SegmentationModelBase):
    """
    RE_DETR (Referring Expression DETR) for object segmentation.
    
    RE_DETR can segment objects based on text descriptions or detect all objects.
    For now, we'll use it to detect and segment all objects in the image.
    """
    def __init__(self, model_path: Optional[str] = None, device: str = 'cuda',
                 conf_threshold: float = 0.3):
        """
        Args:
            model_path: Path to RE_DETR model weights
            device: Device to run model on
            conf_threshold: Confidence threshold for detections
        """
        super().__init__(device)
        self.conf_threshold = conf_threshold
        self.model = None
        
        try:
            # Try to load RE_DETR
            # Note: RE_DETR might need custom installation
            # For now, we'll create a placeholder that can be extended
            try:
                # Attempt to import RE_DETR (this is a placeholder - actual import may differ)
                # RE_DETR is typically from a custom repository
                warnings.warn(
                    "RE_DETR requires custom installation. "
                    "Please install from: https://github.com/Atten4Vis/RE-DETR or similar. "
                    "Using placeholder for now."
                )
            except Exception as e:
                warnings.warn(f"RE_DETR not available: {e}. Using placeholder.")
        except Exception as e:
            warnings.warn(f"Failed to load RE_DETR: {e}. Using placeholder.")
    
    def segment(self, image: torch.Tensor) -> torch.Tensor:
        """
        Generate segmentation mask using RE_DETR.
        
        Args:
            image: RGB image tensor, normalized to [0, 1]
                  Shape: (C, H, W), (batch, C, H, W), or (batch, num_cam, C, H, W)
        
        Returns:
            Segmentation mask tensor with same shape as input
        """
        if self.model is None:
            # Fallback to placeholder
            warnings.warn("RE_DETR model not loaded. Using placeholder segmentation.")
            placeholder = PlaceholderSegmentation(self.device)
            return placeholder.segment(image)
        
        # TODO: Implement actual RE_DETR inference
        # For now, use placeholder
        placeholder = PlaceholderSegmentation(self.device)
        return placeholder.segment(image)


def load_segmentation_model(model_type: str = 'placeholder', 
                           model_path: Optional[str] = None,
                           device: str = 'cuda',
                           **kwargs) -> SegmentationModelBase:
    """
    Load a segmentation model by type.
    
    Args:
        model_type: Type of model ('placeholder', 'fastsam', 're_detr')
        model_path: Path to model weights (optional)
        device: Device to run model on
        **kwargs: Additional model-specific arguments
    
    Returns:
        SegmentationModelBase instance
    """
    model_type = model_type.lower()
    
    if model_type == 'placeholder' or model_type == 'none':
        return PlaceholderSegmentation(device=device)
    elif model_type == 'fastsam':
        return FastSAMSegmentation(
            model_path=model_path,
            device=device,
            conf_threshold=kwargs.get('conf_threshold', 0.25),
            iou_threshold=kwargs.get('iou_threshold', 0.45)
        )
    elif model_type == 're_detr' or model_type == 'redetr':
        return REDETRSegmentation(
            model_path=model_path,
            device=device,
            conf_threshold=kwargs.get('conf_threshold', 0.3)
        )
    else:
        raise ValueError(
            f"Unknown segmentation model type: {model_type}. "
            f"Supported types: 'placeholder', 'fastsam', 're_detr'"
        )
