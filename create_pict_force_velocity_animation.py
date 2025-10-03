#!/usr/bin/env python3
"""
为PICT数据创建Force Component和Velocity Magnitude动画
特点：PICT数据中force是时间变化的
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import os
import glob
from typing import List, Tuple
import argparse
import re

class PictDataAnimator:
    def __init__(self, data_dir: str):
        """
        初始化PICT数据动画创建器
        
        Args:
            data_dir: PICT数据目录路径
        """
        self.data_dir = data_dir
        self.warmup_files = []
        self.main_files = []
        self.all_frames = []
        self.frame_metadata = []
        
        # 动画参数
        self.frame_skip = 20  # 每隔几帧显示一次（在每个文件内部应用）
        
    def load_data_files(self):
        """加载所有PICT数据文件并分类"""
        print("📁 扫描PICT数据文件...")
        
        # 查找warmup文件
        warmup_pattern = os.path.join(self.data_dir, "turbulence_warmup_step*_256x256_index_1.npz")
        self.warmup_files = glob.glob(warmup_pattern)
        
        # 查找主simulation文件
        main_pattern = os.path.join(self.data_dir, "turbulence_step*_256x256_index_1.npz")
        all_main = glob.glob(main_pattern)
        # 排除warmup文件
        self.main_files = [f for f in all_main if 'warmup' not in f]
        
        # 排序文件
        def extract_step_number(filename):
            """从文件名提取步数"""
            match = re.search(r'step(\d+)_', filename)
            return int(match.group(1)) if match else 0
        
        self.warmup_files.sort(key=extract_step_number)
        self.main_files.sort(key=extract_step_number)
        
        # 使用所有文件，不进行文件级采样
        # frame_skip会在每个文件内部应用
        
        print(f"✅ 找到文件:")
        print(f"  Warmup文件: {len(self.warmup_files)} 个")
        print(f"  主simulation文件: {len(self.main_files)} 个")
        print(f"  总计: {len(self.warmup_files) + len(self.main_files)} 个")
        
        if not (self.warmup_files or self.main_files):
            raise ValueError("未找到PICT数据文件!")
    
    def build_frame_index(self):
        """构建所有帧的索引"""
        print("📊 构建帧索引...")
        
        # 先处理warmup文件
        for file_idx, file_path in enumerate(self.warmup_files):
            try:
                data = np.load(file_path)
                
                if 'u' not in data or 'v' not in data:
                    print(f"⚠️ 跳过文件 {os.path.basename(file_path)}: 缺少velocity数据")
                    continue
                
                has_force = 'fu' in data and 'fv' in data
                num_frames = data['u'].shape[0]
                
                for frame_idx in range(0, num_frames, self.frame_skip):
                    self.all_frames.append(('warmup', file_idx, frame_idx))
                    self.frame_metadata.append({
                        'file_path': file_path,
                        'phase': 'warmup',
                        'file_idx': file_idx,
                        'frame_idx': frame_idx,
                        'has_force': has_force,
                        'step_name': os.path.basename(file_path).replace('.npz', ''),
                        'time_idx': len(self.all_frames) - 1
                    })
                
                print(f"  Warmup {file_idx+1}/{len(self.warmup_files)}: {os.path.basename(file_path)} - {num_frames} 帧")
                
            except Exception as e:
                print(f"❌ 加载warmup文件失败 {file_path}: {e}")
                continue
        
        # 处理主simulation文件
        for file_idx, file_path in enumerate(self.main_files):
            try:
                data = np.load(file_path)
                
                if 'u' not in data or 'v' not in data:
                    print(f"⚠️ 跳过文件 {os.path.basename(file_path)}: 缺少velocity数据")
                    continue
                
                has_force = 'fu' in data and 'fv' in data
                num_frames = data['u'].shape[0]
                
                for frame_idx in range(0, num_frames, self.frame_skip):
                    self.all_frames.append(('main', file_idx, frame_idx))
                    self.frame_metadata.append({
                        'file_path': file_path,
                        'phase': 'main',
                        'file_idx': file_idx,
                        'frame_idx': frame_idx,
                        'has_force': has_force,
                        'step_name': os.path.basename(file_path).replace('.npz', ''),
                        'time_idx': len(self.all_frames) - 1
                    })
                
                print(f"  Main {file_idx+1}/{len(self.main_files)}: {os.path.basename(file_path)} - {num_frames} 帧")
                
            except Exception as e:
                print(f"❌ 加载main文件失败 {file_path}: {e}")
                continue
        
        print(f"✅ 总共 {len(self.all_frames)} 个动画帧")
    
    def load_frame_data(self, frame_info: Tuple[str, int, int]):
        """
        加载指定帧的数据
        
        Args:
            frame_info: ('phase', file_idx, frame_idx) 元组
            
        Returns:
            dict: 包含u, v, fu, fv等数据的字典
        """
        phase, file_idx, frame_idx = frame_info
        
        if phase == 'warmup':
            file_path = self.warmup_files[file_idx]
        else:
            file_path = self.main_files[file_idx]
        
        data = np.load(file_path)
        
        result = {
            'u': data['u'][frame_idx],
            'v': data['v'][frame_idx],
            'has_force': False,
            'fu': None,
            'fv': None,
            'time': data.get('time_array', [0])[frame_idx] if 'time_array' in data and frame_idx < len(data['time_array']) else frame_idx
        }
        
        # 检查是否有force数据 - PICT中force是时间变化的
        if 'fu' in data and 'fv' in data:
            result['has_force'] = True
            result['fu'] = data['fu'][frame_idx]  # PICT force是时间变化的
            result['fv'] = data['fv'][frame_idx]  # PICT force是时间变化的
        
        return result
    
    def setup_animation(self):
        """设置动画布局"""
        print("🎬 设置动画布局...")
        
        # 创建figure和subplots
        self.fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(2, 4, figure=self.fig, hspace=0.3, wspace=0.3)
        
        # 创建子图
        self.ax_fu = self.fig.add_subplot(gs[0, 0])       # Force X
        self.ax_fv = self.fig.add_subplot(gs[0, 1])       # Force Y  
        self.ax_force_mag = self.fig.add_subplot(gs[0, 2]) # Force Magnitude
        self.ax_stats = self.fig.add_subplot(gs[0, 3])    # 统计信息
        
        self.ax_u = self.fig.add_subplot(gs[1, 0])         # Velocity X
        self.ax_v = self.fig.add_subplot(gs[1, 1])         # Velocity Y
        self.ax_vel_mag = self.fig.add_subplot(gs[1, 2])   # Velocity Magnitude
        self.ax_profile = self.fig.add_subplot(gs[1, 3])   # Profile comparison
        
        # 设置标题
        self.ax_fu.set_title('Force X (fu) - TIME VARYING', fontsize=12, fontweight='bold', color='red')
        self.ax_fv.set_title('Force Y (fv) - TIME VARYING', fontsize=12, fontweight='bold', color='red')
        self.ax_force_mag.set_title('Force Magnitude', fontsize=12, fontweight='bold')
        self.ax_stats.set_title('Statistics', fontsize=12, fontweight='bold')
        
        self.ax_u.set_title('Velocity X (u)', fontsize=12, fontweight='bold')
        self.ax_v.set_title('Velocity Y (v)', fontsize=12, fontweight='bold')
        self.ax_vel_mag.set_title('Velocity Magnitude', fontsize=12, fontweight='bold')
        self.ax_profile.set_title('Force vs Velocity Profile', fontsize=12, fontweight='bold')
        
        # 关闭坐标轴
        for ax in [self.ax_fu, self.ax_fv, self.ax_force_mag, self.ax_u, self.ax_v, self.ax_vel_mag]:
            ax.set_xticks([])
            ax.set_yticks([])
        
        # 设置main title
        self.fig.suptitle('PICT Data: TIME-VARYING Force Components & Velocity Magnitude Evolution', 
                         fontsize=16, fontweight='bold', color='darkred')
    
    def calculate_global_ranges(self):
        """计算全局数据范围用于一致的colormap"""
        print("📊 计算全局数据范围...")
        
        # 采样一些帧来估计范围 - 减少采样数量以加快速度
        sample_indices = np.linspace(0, len(self.all_frames)-1, min(20, len(self.all_frames)), dtype=int)
        
        u_vals, v_vals, vel_mag_vals = [], [], []
        fu_vals, fv_vals, force_mag_vals = [], [], []
        
        print(f"  采样 {len(sample_indices)} 个帧来估计数据范围...")
        for i, idx in enumerate(sample_indices):
            print(f"\r  进度: {i+1}/{len(sample_indices)}", end="")
            frame_data = self.load_frame_data(self.all_frames[idx])
            
            u, v = frame_data['u'], frame_data['v']
            vel_mag = np.sqrt(u**2 + v**2)
            
            u_vals.extend([np.min(u), np.max(u)])
            v_vals.extend([np.min(v), np.max(v)])
            vel_mag_vals.extend([np.min(vel_mag), np.max(vel_mag)])
            
            if frame_data['has_force']:
                fu, fv = frame_data['fu'], frame_data['fv']
                force_mag = np.sqrt(fu**2 + fv**2)
                
                fu_vals.extend([np.min(fu), np.max(fu)])
                fv_vals.extend([np.min(fv), np.max(fv)])
                force_mag_vals.extend([np.min(force_mag), np.max(force_mag)])
        
        # 设置范围
        self.u_range = (np.min(u_vals), np.max(u_vals))
        self.v_range = (np.min(v_vals), np.max(v_vals))
        self.vel_mag_range = (0, np.max(vel_mag_vals))
        
        if fu_vals:  # 如果有force数据
            self.fu_range = (np.min(fu_vals), np.max(fu_vals))
            self.fv_range = (np.min(fv_vals), np.max(fv_vals))
            self.force_mag_range = (0, np.max(force_mag_vals))
        else:
            self.fu_range = (-1, 1)
            self.fv_range = (-1, 1)
            self.force_mag_range = (0, 1)
        
        print(f"\n  Velocity范围: u={self.u_range}, v={self.v_range}, mag={self.vel_mag_range}")
        print(f"  Force范围: fu={self.fu_range}, fv={self.fv_range}, mag={self.force_mag_range}")
    
    def animate(self, frame_num):
        """动画帧函数"""
        # 清除所有子图
        for ax in [self.ax_fu, self.ax_fv, self.ax_force_mag, self.ax_u, self.ax_v, self.ax_vel_mag, self.ax_profile]:
            ax.clear()
        
        # 重新设置标题
        self.ax_fu.set_title('Force X (fu) - TIME VARYING', fontsize=12, fontweight='bold', color='red')
        self.ax_fv.set_title('Force Y (fv) - TIME VARYING', fontsize=12, fontweight='bold', color='red')
        self.ax_force_mag.set_title('Force Magnitude', fontsize=12, fontweight='bold')
        self.ax_u.set_title('Velocity X (u)', fontsize=12, fontweight='bold')
        self.ax_v.set_title('Velocity Y (v)', fontsize=12, fontweight='bold')
        self.ax_vel_mag.set_title('Velocity Magnitude', fontsize=12, fontweight='bold')
        self.ax_profile.set_title('Force vs Velocity Profile', fontsize=12, fontweight='bold')
        
        # 加载当前帧数据
        frame_data = self.load_frame_data(self.all_frames[frame_num])
        metadata = self.frame_metadata[frame_num]
        
        u, v = frame_data['u'], frame_data['v']
        vel_mag = np.sqrt(u**2 + v**2)
        
        # 显示Force数据（PICT中force是时间变化的）
        if frame_data['has_force']:
            fu, fv = frame_data['fu'], frame_data['fv']
            force_mag = np.sqrt(fu**2 + fv**2)
            
            # Force visualizations with time-varying highlighting
            im1 = self.ax_fu.imshow(fu, cmap='RdBu_r', origin='lower', 
                                   vmin=self.fu_range[0], vmax=self.fu_range[1])
            im2 = self.ax_fv.imshow(fv, cmap='RdBu_r', origin='lower',
                                   vmin=self.fv_range[0], vmax=self.fv_range[1])
            im3 = self.ax_force_mag.imshow(force_mag, cmap='plasma', origin='lower',
                                          vmin=self.force_mag_range[0], vmax=self.force_mag_range[1])
        else:
            # 如果没有force数据，显示占位符
            self.ax_fu.text(0.5, 0.5, 'No Force Data', ha='center', va='center', transform=self.ax_fu.transAxes)
            self.ax_fv.text(0.5, 0.5, 'No Force Data', ha='center', va='center', transform=self.ax_fv.transAxes)
            self.ax_force_mag.text(0.5, 0.5, 'No Force Data', ha='center', va='center', transform=self.ax_force_mag.transAxes)
        
        # Velocity visualizations
        im4 = self.ax_u.imshow(u, cmap='RdBu_r', origin='lower',
                              vmin=self.u_range[0], vmax=self.u_range[1])
        im5 = self.ax_v.imshow(v, cmap='RdBu_r', origin='lower',
                              vmin=self.v_range[0], vmax=self.v_range[1])
        im6 = self.ax_vel_mag.imshow(vel_mag, cmap='viridis', origin='lower',
                                    vmin=self.vel_mag_range[0], vmax=self.vel_mag_range[1])
        
        # Force vs Velocity profile对比
        mid_idx = u.shape[0] // 2
        y_coords = np.linspace(0, 1, u.shape[0])
        
        # Velocity profiles
        self.ax_profile.plot(y_coords, u[:, mid_idx], 'b-', linewidth=2, label='u(y)', alpha=0.8)
        self.ax_profile.plot(y_coords, v[mid_idx, :], 'r-', linewidth=2, label='v(x)', alpha=0.8)
        
        # Force profiles (if available)
        if frame_data['has_force']:
            fu_profile = fu[:, mid_idx]
            fv_profile = fv[mid_idx, :]
            
            # 归一化force以便对比
            fu_norm = fu_profile / np.max(np.abs(fu_profile)) if np.max(np.abs(fu_profile)) > 0 else fu_profile
            fv_norm = fv_profile / np.max(np.abs(fv_profile)) if np.max(np.abs(fv_profile)) > 0 else fv_profile
            
            scale_factor = np.max(np.abs([u[:, mid_idx], v[mid_idx, :]]))
            
            self.ax_profile.plot(y_coords, fu_norm * scale_factor, 'b--', linewidth=2, 
                               label='fu(y) norm', alpha=0.6)
            self.ax_profile.plot(y_coords, fv_norm * scale_factor, 'r--', linewidth=2, 
                               label='fv(x) norm', alpha=0.6)
        
        self.ax_profile.set_xlabel('Position')
        self.ax_profile.set_ylabel('Value')
        self.ax_profile.legend(fontsize=9)
        self.ax_profile.grid(True, alpha=0.3)
        
        # 关闭坐标轴
        for ax in [self.ax_fu, self.ax_fv, self.ax_force_mag, self.ax_u, self.ax_v, self.ax_vel_mag]:
            ax.set_xticks([])
            ax.set_yticks([])
        
        # 统计信息
        self.ax_stats.axis('off')
        
        phase_color = 'blue' if metadata['phase'] == 'warmup' else 'green'
        
        stats_text = f"""Frame: {frame_num+1}/{len(self.all_frames)}
Phase: {metadata['phase'].upper()}
File: {metadata['step_name']}
Frame in file: {metadata['frame_idx']+1}
Time: {frame_data['time']:.3f}

Velocity Stats:
  u: [{np.min(u):.3f}, {np.max(u):.3f}]
  v: [{np.min(v):.3f}, {np.max(v):.3f}]
  |v|: [{np.min(vel_mag):.3f}, {np.max(vel_mag):.3f}]
  RMS: {np.sqrt(np.mean(vel_mag**2)):.3f}"""
        
        if frame_data['has_force']:
            stats_text += f"""

Force Stats (TIME-VARYING):
  fu: [{np.min(fu):.3f}, {np.max(fu):.3f}]
  fv: [{np.min(fv):.3f}, {np.max(fv):.3f}]
  |f|: [{np.min(force_mag):.3f}, {np.max(force_mag):.3f}]
  Force RMS: {np.sqrt(np.mean(force_mag**2)):.3f}
  Force变化: ✓ 动态"""
        else:
            stats_text += "\n\nForce Stats:\n  No force data"
        
        self.ax_stats.text(0.05, 0.95, stats_text, transform=self.ax_stats.transAxes,
                          verticalalignment='top', fontfamily='monospace', fontsize=10,
                          bbox=dict(boxstyle="round,pad=0.3", facecolor=phase_color, alpha=0.2))
        
        # 更新main title
        phase_indicator = "🔥 WARMUP" if metadata['phase'] == 'warmup' else "⚡ MAIN SIM"
        self.fig.suptitle(f'PICT Animation - {phase_indicator} - {metadata["step_name"]} (Frame {frame_num+1}/{len(self.all_frames)})', 
                         fontsize=16, fontweight='bold', color='darkred')
        
        return []
    
    def create_animation(self, output_path: str, fps: int = 10):
        """创建并保存动画"""
        print(f"🎬 创建PICT动画: {len(self.all_frames)} 帧，{fps} FPS")
        
        # 创建动画
        anim = animation.FuncAnimation(
            self.fig, self.animate, frames=len(self.all_frames),
            interval=1000//fps, blit=False, repeat=False
        )
        
        # 保存动画
        print(f"💾 保存动画到: {output_path}")
        
        # 使用更快的编码参数
        writer = animation.FFMpegWriter(
            fps=fps, 
            metadata=dict(artist='PICT Data Animator - Time Varying Forces'),
            bitrate=3000,
            extra_args=['-pix_fmt', 'yuv420p']
        )
        
        anim.save(output_path, writer=writer, progress_callback=lambda i, n: print(f"\r进度: {i+1}/{n} ({(i+1)/n*100:.1f}%)", end=""))
        print(f"\n✅ 动画保存完成!")
        
        return anim

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='创建PICT数据Force+Velocity动画')
    parser.add_argument('--data_dir', type=str, 
                       default='/Users/yiwei/Projects/Python/thesis/JAX-CFD/data/pict_data/kf_256',
                       help='PICT数据目录路径')
    parser.add_argument('--output', type=str, 
                       default='/Users/yiwei/Projects/Python/thesis/JAX-CFD/pict_force_velocity_animation.mp4',
                       help='输出MP4文件路径')
    parser.add_argument('--fps', type=int, default=8, help='动画帧率')
    parser.add_argument('--frame_skip', type=int, default=15, help='帧采样间隔')
    
    args = parser.parse_args()
    
    print("🎬 PICT数据Force+Velocity动画生成器")
    print("🔥 特色：时间变化的Force数据")
    print("="*60)
    print(f"数据目录: {args.data_dir}")
    print(f"输出文件: {args.output}")
    print(f"帧率: {args.fps} FPS")
    print(f"帧间隔: {args.frame_skip}")
    print()
    
    # 创建动画器
    animator = PictDataAnimator(args.data_dir)
    animator.frame_skip = args.frame_skip
    
    try:
        # 加载数据
        animator.load_data_files()
        animator.build_frame_index()
        
        # 设置动画
        animator.calculate_global_ranges()
        animator.setup_animation()
        
        # 创建动画
        anim = animator.create_animation(args.output, args.fps)
        
        # 清理
        plt.close('all')
        
        print(f"\n🎉 成功生成PICT动画: {args.output}")
        print("🔥 特色：展示了PICT数据中时间变化的Force！")
        
    except Exception as e:
        print(f"\n❌ 生成动画时出错: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
