#!/bin/bash
# 运行多分辨率空间涡度相关性分析
# 自动处理warmup和simulation阶段，以2048为参考

echo "🔬 Multi-Resolution Spatial Vorticity Correlation Analysis"
echo "========================================================"

# 检查基础路径
BASE_DIR="/Volumes/T7"
if [ ! -d "$BASE_DIR" ]; then
    echo "❌ Base directory not found: $BASE_DIR"
    exit 1
fi

echo "📁 Base directory: $BASE_DIR"

# 设置参数
REFERENCE_RES=2048
COMPARISON_RESOLUTIONS="64 128 256 512 1024"
MAX_STEPS=15000 # 包含所有warmup数据 (包括warmup_final)
SAVE_DIR="multiresolution_correlation_results_$(date +%Y%m%d_%H%M%S)"

# 检查参考分辨率文件夹
REF_DIR="${BASE_DIR}/${REFERENCE_RES}"
if [ ! -d "$REF_DIR" ]; then
    echo "❌ Reference directory not found: $REF_DIR"
    exit 1
fi

echo "✅ Reference resolution: ${REFERENCE_RES}x${REFERENCE_RES} (${REF_DIR})"

# 检查比较分辨率文件夹
echo ""
echo "📊 Checking comparison resolutions:"
VALID_RESOLUTIONS=""
for res in $COMPARISON_RESOLUTIONS; do
    res_dir="${BASE_DIR}/${res}"
    if [ -d "$res_dir" ]; then
        echo "✅ ${res}x${res}: $res_dir"
        VALID_RESOLUTIONS="$VALID_RESOLUTIONS $res"
    else
        echo "⚠️  ${res}x${res}: directory not found ($res_dir)"
    fi
done

if [ -z "$VALID_RESOLUTIONS" ]; then
    echo "❌ No valid comparison resolutions found!"
    exit 1
fi

echo ""
echo "🚀 Starting analysis..."
echo "Reference: ${REFERENCE_RES}x${REFERENCE_RES}"
echo "Comparisons:$VALID_RESOLUTIONS"
echo "Max steps: $MAX_STEPS"
echo "Save directory: $SAVE_DIR"
echo ""

# 运行Python分析
python multiresolution_spatial_correlation_analysis.py \
    --base_dir "$BASE_DIR" \
    --reference_resolution $REFERENCE_RES \
    --comparison_resolutions $VALID_RESOLUTIONS \
    --max_steps $MAX_STEPS \
    --save_dir "$SAVE_DIR"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Analysis completed successfully!"
    echo "📁 Results saved in: $SAVE_DIR"
    
    if [ -d "$SAVE_DIR" ]; then
        echo ""
        echo "🔍 Generated files:"
        ls -la "$SAVE_DIR"/
        
        # 检查是否生成了图片
        if [ -f "$SAVE_DIR"/multiresolution_spatial_correlation_with_phases.png ]; then
            echo ""
            echo "🎨 Visualization created:"
            echo "   $SAVE_DIR/multiresolution_spatial_correlation_with_phases.png"
            echo ""
            echo "📊 The plot shows:"
            echo "   - Blue background: Warmup phase"
            echo "   - Orange background: Simulation phase"
            echo "   - Each line represents correlation with 2048x2048 reference"
        fi
    fi
else
    echo ""
    echo "❌ Analysis failed!"
    exit 1
fi

echo ""
echo "🎯 Analysis complete! The plot shows spatial vorticity correlation"
echo "   between each resolution and the 2048x2048 reference, with"
echo "   background colors distinguishing warmup and simulation phases."
