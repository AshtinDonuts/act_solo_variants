# Segmentation Models for SegACT

SegACT supports multiple segmentation models that can be switched via configuration.

## Supported Models

### 1. Placeholder (Default)
- **Type**: `placeholder`
- **Description**: Simple grayscale conversion of RGB images
- **Use case**: Baseline, fallback, or when no segmentation model is available
- **No dependencies required**

### 2. FastSAM
- **Type**: `fastsam`
- **Description**: Fast Segment Anything Model via ultralytics
- **Dependencies**: `pip install ultralytics`
- **Model weights**: Automatically downloads `FastSAM-x.pt` if not provided
- **Parameters**:
  - `conf_threshold` (default: 0.25): Confidence threshold for detections
  - `iou_threshold` (default: 0.45): IoU threshold for NMS
  - `model_path` (optional): Path to custom model weights

### 3. RE_DETR
- **Type**: `re_detr` or `redetr`
- **Description**: Referring Expression DETR for object segmentation
- **Status**: Placeholder implementation (requires custom installation)
- **Note**: RE_DETR typically requires installation from custom repositories
- **Parameters**:
  - `conf_threshold` (default: 0.3): Confidence threshold for detections
  - `model_path` (optional): Path to model weights

## Usage

### Command Line

```bash
# Use placeholder (default)
python act/imitate_episodes.py --policy_class SegACT --segmentation_model placeholder ...

# Use FastSAM
python act/imitate_episodes.py --policy_class SegACT --segmentation_model fastsam ...

# Use FastSAM with custom model path
python act/imitate_episodes.py --policy_class SegACT --segmentation_model fastsam \
    --segmentation_model_path /path/to/FastSAM-x.pt ...
```

### Configuration

In `imitate_episodes.py`, you can pass segmentation model configuration:

```python
policy_config = {
    ...
    'segmentation_model': 'fastsam',  # or 'placeholder', 're_detr'
    'segmentation_model_path': None,  # Optional path to model weights
    'segmentation_model_kwargs': {
        'conf_threshold': 0.25,  # For FastSAM
        'iou_threshold': 0.45,   # For FastSAM
    }
}
```

## Implementation Details

### Architecture

- **Base class**: `SegmentationModelBase` in `segmentation_models.py`
- **Factory function**: `load_segmentation_model()` loads models by type
- **Integration**: Models are loaded in `SegACT_Policy.__init__()` and used in `_get_segmentation_mask()`

### Adding New Models

To add a new segmentation model:

1. Create a class inheriting from `SegmentationModelBase`
2. Implement the `segment()` method
3. Add the model type to `load_segmentation_model()`
4. Update argument parser in `main.py` and `imitate_episodes.py`

Example:

```python
class MySegmentationModel(SegmentationModelBase):
    def __init__(self, device='cuda', **kwargs):
        super().__init__(device)
        # Initialize your model
    
    def segment(self, image: torch.Tensor) -> torch.Tensor:
        # Implement segmentation logic
        # Input: (C, H, W), (batch, C, H, W), or (batch, num_cam, C, H, W)
        # Output: Same shape as input
        return segmentation_mask
```

## Notes

- All models should handle images normalized to [0, 1]
- Models should support batch processing and multi-camera inputs
- Models fall back to placeholder if loading fails
- FastSAM automatically downloads weights on first use if not provided
