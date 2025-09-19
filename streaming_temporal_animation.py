#!/usr/bin/env python3
"""
流式时间序列动画：不一次性加载所有帧，按需读取
按 warmup → 主要模拟 → final 的顺序
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import glob
import argparse
from pathlib import Path
import re

class StreamingTemporalAnimator:
    def __init__(self, data_dir, output_dir="streaming_temporal_animations"):
        self.data_dir = data_dir
        self.output_dir = output_dir
        Path(self.output_dir).mkdir(exist_ok=True)
        
        # 构建文件列表和帧索引，但不加载数据
        self.build_frame_index()
        
    def build_frame_index(self):
        """构建帧索引，不加载实际数据"""
        print("构建帧索引...")
        
        # 获取所有文件
        all_files = glob.glob(os.path.join(self.data_dir, "*.npz"))
        
        # 分类文件
        warmup_files = []
        main_files = []
        final_files = []
        
        for f in all_files:
            filename = os.path.basename(f)
            
            if 'warmup' in filename and 'step' in filename:
                step_match = re.search(r'step(\d+)_', filename)
                if step_match:
                    step_num = int(step_match.group(1))
                    warmup_files.append((step_num, f))
            
            elif 'step' in filename and 'warmup' not in filename:
                step_match = re.search(r'step(\d+)_', filename)
                if step_match:
                    step_num = int(step_match.group(1))
                    main_files.append((step_num, f))
            
            elif 'final' in filename:
                final_files.append(f)
        
        # 排序
        warmup_files = sorted(warmup_files, key=lambda x: x[0])
        main_files = sorted(main_files, key=lambda x: x[0])
        
        print(f"文件统计:")
        print(f"  Warmup: {len(warmup_files)} 个文件")
        print(f"  主要模拟: {len(main_files)} 个文件") 
        print(f"  Final: {len(final_files)} 个文件")
        
        # 构建帧索引 (file_path, frame_index, phase, step_num)
        self.frame_index = []
        cumulative_time = 0.0
        
        # 1. Warmup阶段
        print(f"\n构建Warmup阶段索引:")
        warmup_frame_count = 0
        for step_num, filepath in warmup_files:
            n_frames = self.get_frame_count(filepath)
            if n_frames > 0:
                for frame_idx in range(n_frames):
                    self.frame_index.append({
                        'file_path': filepath,
                        'frame_idx': frame_idx,
                        'phase': 'Warmup',
                        'step': step_num,
                        'global_time': cumulative_time + frame_idx * 0.000018
                    })
                warmup_frame_count += n_frames
                cumulative_time += n_frames * 0.000018
                
        print(f"  Warmup帧索引: {warmup_frame_count} 帧")
        
        # 2. 主要模拟阶段
        print(f"\n构建主要模拟阶段索引:")
        main_frame_count = 0
        for step_num, filepath in main_files:
            n_frames = self.get_frame_count(filepath)
            if n_frames > 0:
                for frame_idx in range(n_frames):
                    self.frame_index.append({
                        'file_path': filepath,
                        'frame_idx': frame_idx,
                        'phase': 'Main',
                        'step': step_num,
                        'global_time': cumulative_time + frame_idx * 0.000018
                    })
                main_frame_count += n_frames
                cumulative_time += n_frames * 0.000018
                
        print(f"  主要模拟帧索引: {main_frame_count} 帧")
        
        # 3. Final阶段
        print(f"\n构建Final阶段索引:")
        final_frame_count = 0
        for i, filepath in enumerate(final_files):
            n_frames = self.get_frame_count(filepath)
            if n_frames > 0:
                for frame_idx in range(n_frames):
                    self.frame_index.append({
                        'file_path': filepath,
                        'frame_idx': frame_idx,
                        'phase': f'Final-{i+1}',
                        'step': 99999,
                        'global_time': cumulative_time + frame_idx * 0.000018
                    })
                final_frame_count += n_frames
                cumulative_time += n_frames * 0.000018
                
        print(f"  Final帧索引: {final_frame_count} 帧")
        print(f"\n总帧索引: {len(self.frame_index)} 帧")
        
        # 计算全局范围（采样）
        self.calculate_global_ranges_sampled()
        
    def get_frame_count(self, filepath):
        """获取文件的帧数，不加载实际数据"""
        try:
            data = np.load(filepath)
            if 'u' in data.keys():
                return data['u'].shape[0]
            return 0
        except:
            return 0
    
    def load_frame_data(self, frame_info):
        """按需加载单帧数据"""
        try:
            data = np.load(frame_info['file_path'])
            frame_idx = frame_info['frame_idx']
            
            has_force = 'fu' in data.keys() and 'fv' in data.keys()
            
            frame_data = {
                'u': data['u'][frame_idx],
                'v': data['v'][frame_idx], 
                'fu': data['fu'][frame_idx] if has_force else np.zeros_like(data['u'][frame_idx]),
                'fv': data['fv'][frame_idx] if has_force else np.zeros_like(data['v'][frame_idx]),
                'has_force': has_force,
                'phase': frame_info['phase'],
                'step': frame_info['step'],
                'frame': frame_info['frame_idx'],
                'time': frame_info['global_time']
            }
            
            return frame_data
            
        except Exception as e:
            print(f"错误加载帧数据: {e}")
            return None
    
    def calculate_global_ranges_sampled(self):
        """通过采样计算全局范围"""
        print("计算全局范围（采样）...")
        
        # 每1000帧采样1个
        sample_indices = range(0, len(self.frame_index), 1000)
        
        all_vel_mag = []
        all_force_mag = []
        all_fu = []
        all_fv = []
        
        for i in sample_indices:
            frame_data = self.load_frame_data(self.frame_index[i])
            if frame_data:
                vel_mag = np.sqrt(frame_data['u']**2 + frame_data['v']**2)
                force_mag = np.sqrt(frame_data['fu']**2 + frame_data['fv']**2)
                
                all_vel_mag.append(np.max(vel_mag))
                all_force_mag.append(np.max(force_mag))
                all_fu.append(np.max(np.abs(frame_data['fu'])))
                all_fv.append(np.max(np.abs(frame_data['fv'])))
        
        self.vel_max = max(all_vel_mag) if all_vel_mag else 1.0
        self.force_max = max(all_force_mag) if all_force_mag else 1.0
        self.fu_max = max(all_fu) if all_fu else 1.0
        self.fv_max = max(all_fv) if all_fv else 1.0
        
        print(f"  采样 {len(sample_indices)} 帧计算范围")
        print(f"  Velocity magnitude max: {self.vel_max:.6f}")
        print(f"  Force magnitude max: {self.force_max:.6f}")
    
    def create_streaming_animation(self, fps=15, frame_skip=1):
        """创建流式动画"""
        print(f"创建流式动画...")
        
        # 应用帧跳跃
        frame_indices_to_use = list(range(0, len(self.frame_index), frame_skip))
        print(f"使用 {len(frame_indices_to_use)} 帧 (每 {frame_skip} 帧取1帧)")
        
        if len(frame_indices_to_use) == 0:
            print("错误：没有可用帧")
            return None
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 14))
        
        # 加载第一帧进行初始化
        first_frame_data = self.load_frame_data(self.frame_index[frame_indices_to_use[0]])
        
        im1 = ax1.imshow(first_frame_data['fu'], cmap='RdBu_r', origin='lower',
                        vmin=-self.fu_max, vmax=self.fu_max)
        im2 = ax2.imshow(first_frame_data['fv'], cmap='RdBu_r', origin='lower',
                        vmin=-self.fv_max, vmax=self.fv_max)
        
        vel_mag = np.sqrt(first_frame_data['u']**2 + first_frame_data['v']**2)
        force_mag = np.sqrt(first_frame_data['fu']**2 + first_frame_data['fv']**2)
        
        im3 = ax3.imshow(vel_mag, cmap='viridis', origin='lower',
                        vmin=0, vmax=self.vel_max)
        im4 = ax4.imshow(force_mag, cmap='plasma', origin='lower',
                        vmin=0, vmax=self.force_max)
        
        # 设置标题和标签
        ax1.set_title('Force X-Component (fu)', fontsize=14)
        ax2.set_title('Force Y-Component (fv)', fontsize=14)
        ax3.set_title('Velocity Magnitude', fontsize=14)
        ax4.set_title('Force Magnitude', fontsize=14)
        
        for ax in [ax1, ax2, ax3, ax4]:
            ax.set_xlabel('X', fontsize=12)
            ax.set_ylabel('Y', fontsize=12)
        
        # 添加颜色条
        plt.colorbar(im1, ax=ax1, shrink=0.8)
        plt.colorbar(im2, ax=ax2, shrink=0.8)
        plt.colorbar(im3, ax=ax3, shrink=0.8)
        plt.colorbar(im4, ax=ax4, shrink=0.8)
        
        # 总标题
        title_text = fig.suptitle('', fontsize=16, y=0.95)
        
        def animate(anim_frame_idx):
            # 动态加载当前帧数据
            global_frame_idx = frame_indices_to_use[anim_frame_idx]
            frame_data = self.load_frame_data(self.frame_index[global_frame_idx])
            
            if frame_data is None:
                return [im1, im2, im3, im4, title_text]
            
            # 计算导出量
            vel_mag = np.sqrt(frame_data['u']**2 + frame_data['v']**2)
            force_mag = np.sqrt(frame_data['fu']**2 + frame_data['fv']**2)
            
            # 更新图像
            im1.set_array(frame_data['fu'])
            im2.set_array(frame_data['fv'])
            im3.set_array(vel_mag)
            im4.set_array(force_mag)
            
            # 更新标题
            phase = frame_data['phase']
            step = frame_data['step']
            frame_num = frame_data['frame']
            time_val = frame_data['time']
            force_status = "✓" if frame_data['has_force'] else "✗"
            
            title = f'{phase} - Step {step} Frame {frame_num} - Time: {time_val:.6f} - Force: {force_status}'
            title_text.set_text(title)
            
            # 打印进度
            if anim_frame_idx % 10 == 0:
                progress = (anim_frame_idx + 1) / len(frame_indices_to_use) * 100
                print(f"  进度: {progress:.1f}% ({anim_frame_idx+1}/{len(frame_indices_to_use)})")
            
            return [im1, im2, im3, im4, title_text]
        
        # 创建动画
        anim = animation.FuncAnimation(fig, animate, frames=len(frame_indices_to_use),
                                     interval=1000//fps, blit=False, repeat=True)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.92])
        
        # 保存动画
        output_file_gif = os.path.join(self.output_dir, f'streaming_temporal_sequence_skip{frame_skip}.gif')
        print(f"保存GIF动画: {output_file_gif}")
        anim.save(output_file_gif, writer='pillow', fps=fps)
        
        # 尝试保存MP4
        try:
            output_file_mp4 = os.path.join(self.output_dir, f'streaming_temporal_sequence_skip{frame_skip}.mp4')
            print(f"尝试保存MP4动画: {output_file_mp4}")
            anim.save(output_file_mp4, writer='ffmpeg', fps=fps, dpi=100,
                     extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
            print("MP4保存成功!")
        except Exception as e:
            print(f"MP4保存失败: {e}")
        
        plt.close()
        return output_file_gif

def main():
    parser = argparse.ArgumentParser(description='创建流式时间序列动画')
    parser.add_argument('--data_dir', 
                       default='/Users/yiwei/Projects/Python/thesis/JAX-CFD/data/pict_data/kf_256',
                       help='包含完整序列的数据目录')
    parser.add_argument('--output_dir', default='streaming_temporal_animations',
                       help='输出目录')
    parser.add_argument('--fps', type=int, default=15,
                       help='动画帧率')
    parser.add_argument('--frame_skip', type=int, default=100,
                       help='每隔多少个时间步取一帧')
    
    args = parser.parse_args()
    
    # 创建动画器
    animator = StreamingTemporalAnimator(args.data_dir, args.output_dir)
    
    print(f"\n准备创建流式动画: {len(animator.frame_index)} 个时间步")
    
    # 创建动画
    output_file = animator.create_streaming_animation(
        fps=args.fps, frame_skip=args.frame_skip)
    
    if output_file:
        print(f"\n完成! 动画文件: {output_file}")
    else:
        print("\n动画创建失败!")

if __name__ == "__main__":
    main()
