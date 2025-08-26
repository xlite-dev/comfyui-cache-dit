"""
ComfyUI 缓存加速节点

这个文件定义了 ComfyUI 的自定义节点，用户可以通过这些节点在工作流中应用缓存加速。
基于验证有效的简单缓存逻辑。
"""

from .cache_engine import patch_model_simple, get_simple_stats


class CacheDitAccelerateNode:
    """
    CacheDit 加速节点
    
    将缓存加速应用到 ComfyUI 模型，实现 2x+ 推理加速
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("加速模型",)
    FUNCTION = "accelerate_model"
    CATEGORY = "CacheDit"

    def accelerate_model(self, model):
        """
        应用缓存加速到模型
        
        Args:
            model: 输入的 ComfyUI 模型
            
        Returns:
            tuple: (加速后的模型,)
        """
        print("\n🚀 应用 CacheDit 加速...")
        
        # 应用缓存补丁
        accelerated_model = patch_model_simple(model)
        
        print("✓ CacheDit 加速已应用")
        return (accelerated_model,)


class CacheDitStatsNode:
    """
    CacheDit 统计节点
    
    显示缓存统计信息，包括命中率和预期加速比
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trigger": ("*",),  # 接受任何类型作为触发器
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("统计信息",)
    FUNCTION = "get_stats"
    CATEGORY = "CacheDit"

    def get_stats(self, trigger):
        """
        获取缓存统计信息
        
        Args:
            trigger: 触发器（任何值）
            
        Returns:
            tuple: (统计信息字符串,)
        """
        stats = get_simple_stats()
        return (stats,)


# 节点映射
NODE_CLASS_MAPPINGS = {
    "CacheDitAccelerate": CacheDitAccelerateNode,
    "CacheDitStats": CacheDitStatsNode,
}

# 节点显示名称映射
NODE_DISPLAY_NAME_MAPPINGS = {
    "CacheDitAccelerate": "CacheDit 模型加速",
    "CacheDitStats": "CacheDit 统计信息",
}