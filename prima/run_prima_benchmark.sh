#!/bin/bash

# 로그 디렉토리 생성
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

# 테스트할 모델 (GGUF 파일명)
MODELS=("qwq-32b-q4_k_m.gguf" "qwq-32b-q6_k.gguf")
CONTEXT_LENGTHS=(1024 2048)
MAX_TOKENS=(256 512)

# 타임스탬프
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# 전체 결과 로그 파일
SUMMARY_LOG="$LOG_DIR/prima_benchmark_summary_${TIMESTAMP}.log"

echo "================================" | tee "$SUMMARY_LOG"
echo "PRIMA.cpp Benchmark Started" | tee -a "$SUMMARY_LOG"
echo "Timestamp: $TIMESTAMP" | tee -a "$SUMMARY_LOG"
echo "================================" | tee -a "$SUMMARY_LOG"
echo "" | tee -a "$SUMMARY_LOG"

# 카운터
TOTAL=0
SUCCESS=0
FAILED=0

# 모든 조합에 대해 반복
for model in "${MODELS[@]}"; do
    for context in "${CONTEXT_LENGTHS[@]}"; do
        for tokens in "${MAX_TOKENS[@]}"; do
            TOTAL=$((TOTAL + 1))
            
            # 로그 파일명 생성
            MODEL_NAME=$(echo "$model" | sed 's/\.gguf//g')
            LOG_FILE="$LOG_DIR/${MODEL_NAME}_ctx${context}_tok${tokens}_${TIMESTAMP}.log"
            
            # 실행 정보 출력
            echo "Running: Model=$model, Context=$context, MaxTokens=$tokens" | tee -a "$SUMMARY_LOG"
            echo "Log file: $LOG_FILE" | tee -a "$SUMMARY_LOG"
            
            # 명령 실행 (download 폴더에 모델이 있다고 가정)
            if ./llama-cli -m "download/$model" -c "$context" -p "what is edge AI?" -n "$tokens" -ngl 30 \
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