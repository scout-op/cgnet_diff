"""
检查数据文件，找到中心线数据的位置
"""

import pickle
import sys

def inspect_pkl(file_path):
    """检查pkl文件内容"""
    print(f"\n{'='*70}")
    print(f"检查文件: {file_path}")
    print('='*70)
    
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"\n数据类型: {type(data)}")
        
        if isinstance(data, dict):
            print(f"字典键: {list(data.keys())}")
            
            if 'infos' in data:
                infos = data['infos']
                print(f"\ninfos数量: {len(infos)}")
                
                if len(infos) > 0:
                    sample = infos[0]
                    print(f"\n第一个样本的键:")
                    for key in sample.keys():
                        value = sample[key]
                        print(f"  {key}: {type(value)}")
                        
                        if key == 'gt_bboxes_3d' and value is not None:
                            print(f"    ✅ 找到gt_bboxes_3d!")
                            print(f"    类型: {type(value)}")
                            if hasattr(value, '__dict__'):
                                print(f"    属性: {list(vars(value).keys())}")
                            if hasattr(value, 'instance_list'):
                                print(f"    instance_list长度: {len(value.instance_list)}")
                                if len(value.instance_list) > 0:
                                    print(f"    第一条线类型: {type(value.instance_list[0])}")
                                    print(f"    第一条线: {value.instance_list[0]}")
        
        elif isinstance(data, list):
            print(f"列表长度: {len(data)}")
            if len(data) > 0:
                print(f"第一个元素类型: {type(data[0])}")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    files_to_check = [
        'data/nuscenes/nuscenes_centerline_infos_train.pkl',
        'data/nuscenes/nuscenes_centerline_infos_val.pkl',
        'data/nuscenes/anns/nuscenes_infos_temporal_train.pkl',
        'data/nuscenes/anns/nuscenes_map_anns_val_centerline.json',
    ]
    
    for file_path in files_to_check:
        inspect_pkl(file_path)
    
    print("\n" + "="*70)
    print("检查完成！")
    print("="*70)
