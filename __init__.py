"""
ComfyUI CacheDit 加速插件

基于简单有效的缓存逻辑实现 diffusion 模型推理加速
在 FLUX 等模型上测试有效，能实现约 2x 加速效果
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']