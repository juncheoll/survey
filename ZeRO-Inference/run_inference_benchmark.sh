#!/bin/bash

# 로그 디렉토리 생성
LOG_DIR="/logs"
mkdir -p "$LOG_DIR"

# 테스트할 모델과 배치 크기
MODELS=("facebook/opt-1.3b" "facebook/opt-2.7b" "facebook/opt-6.7b" "facebook/opt-13b")
BATCH_SIZES=(4 8 12 16 32 64)
QUANT_BITS=(16 8 4)

# 타임스탬프
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# 전체 결과 로그 파일
SUMMARY_LOG="$LOG_DIR/benchmark_summary_${TIMESTAMP}.log"

echo "================================" | tee "$SUMMARY_LOG"
echo "ZeRO-Inference Benchmark Started" | tee -a "$SUMMARY_LOG"
echo "Timestamp: $TIMESTAMP" | tee -a "$SUMMARY_LOG"
echo "================================" | tee -a "$SUMMARY_LOG"
echo "" | tee -a "$SUMMARY_LOG"

# 카운터
TOTAL=0
SUCCESS=0
FAILED=0

# 모든 조합에 대해 반복
for model in "${MODELS[@]}"; do
    for batch_size in "${BATCH_SIZES[@]}"; do
        for quant_bits in "${QUANT_BITS[@]}"; do
            TOTAL=$((TOTAL + 1))
            
            # 로그 파일명 생성
            MODEL_NAME=$(echo "$model" | sed 's/\//_/g')
            LOG_FILE="$LOG_DIR/${MODEL_NAME}_batch${batch_size}_quant${quant_bits}_${TIMESTAMP}.log"
            
            # 실행 정보 출력
            echo "Running: Model=$model, Batch Size=$batch_size, QuantBits=$quant_bits" | tee -a "$SUMMARY_LOG"
            echo "Log file: $LOG_FILE" | tee -a "$SUMMARY_LOG"
            
            # quant bits 옵션 설정 (16은 기본값: 옵션 없음)
            QUANT_ARGS=()
            if [ "$quant_bits" != "16" ]; then
                QUANT_ARGS=("--quant_bits" "$quant_bits")
            fi
            
            # 명령 실행
            if python3 run_model.py \
                --model "$model" \
                --batch-size "$batch_size" \
                --prompt-len 512 \
                --gen-len 32 \
                --cpu-offload \
                --kv-offload \
                "${QUANT_ARGS[@]}" \
                > "$LOG_FILE" 2>&1; then
                echo "✓ SUCCESS" | tee -a "$SUMMARY_LOG"
                SUCCESS=$((SUCCESS + 1))
            else
                echo "✗ FAILED" | tee -a "$SUMMARY_LOG"
                FAILED=$((FAILED + 1))
            fi
            
            echo "---" | tee -a "$SUMMARY_LOG"
            echo "" | tee -a "$SUMMARY_LOG"
        done
    done
done

# 최종 결과
echo "================================" | tee -a "$SUMMARY_LOG"
echo "Benchmark Completed" | tee -a "$SUMMARY_LOG"
echo "Total: $TOTAL | Success: $SUCCESS | Failed: $FAILED" | tee -a "$SUMMARY_LOG"
echo "================================" | tee -a "$SUMMARY_LOG"
