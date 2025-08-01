#!/usr/bin/env python3
"""
测试设备修复的简单脚本
"""

import torch

def test_device_handling():
    """测试设备处理逻辑"""
    print("=== 测试设备处理逻辑 ===")
    
    # 模拟模型包装器
    class MockModelWrapper:
        def __init__(self):
            # 模拟模型在GPU上
            self.model = type('obj', (object,), {
                'device': torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
            })()
            
            # 模拟参数
            self.model.parameters = lambda: [torch.randn(10, 10).to(self.model.device)]
        
        def get_logits_and_route_info(self, input_ids):
            # 检查设备
            if hasattr(self.model, 'device'):
                device = self.model.device
            else:
                device = next(self.model.parameters()).device
            
            print(f"模型设备: {device}")
            print(f"输入设备: {input_ids.device}")
            
            # 移动到正确设备
            input_ids = input_ids.to(device)
            print(f"移动后输入设备: {input_ids.device}")
            
            # 模拟返回
            return torch.randn(1, 50257).to(device), 0, 0.8
        
        def get_logits_blackbox(self, input_ids):
            # 检查设备
            if hasattr(self.model, 'device'):
                device = self.model.device
            else:
                device = next(self.model.parameters()).device
            
            # 移动到正确设备
            input_ids = input_ids.to(device)
            
            # 模拟返回
            return torch.randn(1, 50257).to(device)
    
    # 测试
    wrapper = MockModelWrapper()
    
    # 创建CPU上的输入
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    print(f"原始输入设备: {input_ids.device}")
    
    try:
        # 测试灰盒方法
        logits, expert_idx, confidence = wrapper.get_logits_and_route_info(input_ids)
        print(f"✓ 灰盒方法设备处理成功")
        
        # 测试黑盒方法
        logits = wrapper.get_logits_blackbox(input_ids)
        print(f"✓ 黑盒方法设备处理成功")
        
        return True
        
    except Exception as e:
        print(f"✗ 设备处理失败: {e}")
        return False

def test_device_detection():
    """测试设备检测逻辑"""
    print("\n=== 测试设备检测逻辑 ===")
    
    # 测试不同的模型配置
    test_cases = [
        {
            'name': '有device属性的模型',
            'model': type('obj', (object,), {'device': torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')})()
        },
        {
            'name': '有参数的模型',
            'model': type('obj', (object,), {
                'parameters': lambda: [torch.randn(10, 10).to(torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu'))]
            })()
        }
    ]
    
    for case in test_cases:
        print(f"\n测试: {case['name']}")
        model = case['model']
        
        try:
            if hasattr(model, 'device'):
                device = model.device
                print(f"使用model.device: {device}")
            else:
                device = next(model.parameters()).device
                print(f"使用参数设备: {device}")
            
            print(f"✓ 设备检测成功: {device}")
            
        except Exception as e:
            print(f"✗ 设备检测失败: {e}")
            return False
    
    return True

if __name__ == "__main__":
    success1 = test_device_handling()
    success2 = test_device_detection()
    
    if success1 and success2:
        print("\n✓ 所有设备修复测试通过！")
    else:
        print("\n✗ 部分测试失败，需要进一步修复。") 