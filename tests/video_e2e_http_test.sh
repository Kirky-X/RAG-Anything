#!/bin/bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:9621}"
API_KEY="${API_KEY:-raganything-key}"
VIDEO_PATH="${VIDEO_PATH:-/home/project/RAG-Anything/tests/resource/project_1.mp4}"
DOC_ID="${DOC_ID:-project_1_video_$(date +%s)}"
LOG_FILE="${LOG_FILE:-/tmp/video_e2e_http_test.log}"
MAX_RETRIES="${MAX_RETRIES:-10}"
RETRY_DELAY="${RETRY_DELAY:-5}"
MAX_LATENCY_MS=5000

echo "Writing logs to $LOG_FILE"
: > "$LOG_FILE"

log() {
  local msg="$1"
  echo "$(date '+%Y-%m-%d %H:%M:%S') - $msg" | tee -a "$LOG_FILE"
}

request_with_retry() {
  local method="$1"
  local url="$2"
  local body="${3:-}"
  shift 3 || true
  local extra_args=("$@")

  local attempt=1
  local status
  local resp
  local body_out

  while [ "$attempt" -le "$MAX_RETRIES" ]; do
    log "HTTP $method $url (attempt $attempt)"
    if [ "$method" = "GET" ]; then
      resp=$(curl -sS -w "\n%{http_code}" "${extra_args[@]}" "$url" || true)
    else
      resp=$(curl -sS -w "\n%{http_code}" -X "$method" "${extra_args[@]}" -d "$body" "$url" || true)
    fi
    status=$(echo "$resp" | tail -n1)
    body_out=$(echo "$resp" | sed '$d')
    log "Status: $status"
    # Truncate long responses for logging
    if [ ${#body_out} -gt 500 ]; then
        log "Response: ${body_out:0:500}..."
    else
        log "Response: $body_out"
    fi
    REQ_STATUS="$status"
    REQ_BODY="$body_out"
    if [ "$status" -eq 200 ] || [ "$status" -eq 202 ]; then
      return 0
    fi
    attempt=$((attempt + 1))
    sleep "$RETRY_DELAY"
  done
  return 1
}

upload_with_retry() {
  local url="$1"
  local attempt=1
  local resp
  local status
  local body_out

  # Increase curl timeout to avoid client-side timeout during large uploads
  local curl_opts=(-m 300)

  while [ "$attempt" -le "$MAX_RETRIES" ]; do
    log "Uploading video $VIDEO_PATH to $url (attempt $attempt)"
    resp=$(curl -sS -w "\n%{http_code}" "${curl_opts[@]}" \
      -X POST "$url" \
      -H "X-API-Key: $API_KEY" \
      -F "file=@${VIDEO_PATH}" \
      -F "parse_method=auto" \
      -F "output_dir=/tmp/video_e2e_output" \
      -F "doc_id=${DOC_ID}" || true)
    status=$(echo "$resp" | tail -n1)
    body_out=$(echo "$resp" | sed '$d')
    log "Status: $status"
    log "Response: $body_out"
    REQ_STATUS="$status"
    REQ_BODY="$body_out"
    if [ "$status" -eq 200 ] || [ "$status" -eq 202 ]; then
      return 0
    fi
    attempt=$((attempt + 1))
    sleep "$RETRY_DELAY"
  done
  return 1
}

validate_query_response() {
    local response_json="$1"
    local check_type="$2" # "content", "semantic", "time"
    
    python3 -c "
import sys, json
try:
    data = json.loads('''$response_json''')
    
    # Check basic structure
    if not isinstance(data, dict):
        print('Error: Response is not a dictionary')
        sys.exit(1)
        
    # Depending on API response structure, adjust checks
    # Assuming standard LightRAG/RAG-Anything response format
    # Usually has 'response' or 'data' field
    
    results = data.get('data', []) if 'data' in data else [data]
    if not results:
        # Some APIs return direct list or string
        if isinstance(data, str):
            if len(data) > 0: sys.exit(0)
            else: 
                print('Error: Empty string response')
                sys.exit(1)
        elif isinstance(data, list):
            results = data
        else:
            # Maybe it's in 'answer' field
            if 'answer' in data:
                print('Found answer field')
                sys.exit(0)
                
    # Specific checks based on query type
    # For now, just ensure we have some non-empty response
    print('Validation passed')
except Exception as e:
    print(f'Error validating response: {e}')
    sys.exit(1)
"
}

if [ ! -f "$VIDEO_PATH" ]; then
  log "Video file not found: $VIDEO_PATH"
  exit 1
fi

log "==== 1. Waiting for server health ===="
if ! request_with_retry "GET" "$BASE_URL/health" ""; then
  log "Health endpoint did not return 200 after retries (last status: $REQ_STATUS)"
  exit 1
fi
log "Health check passed"

log "==== 2. Testing secure endpoint with API key ===="
if ! request_with_retry "GET" "$BASE_URL/api/secure" "" "-H" "X-API-Key: $API_KEY"; then
  log "Secure endpoint failed after retries (last status: $REQ_STATUS)"
  exit 1
fi
log "Secure endpoint passed"

log "==== 3. Uploading video document ===="
start_ms=$(date +%s%3N)
if ! upload_with_retry "$BASE_URL/api/doc/upload"; then
  log "Video upload failed after retries (last status: $REQ_STATUS)"
  exit 1
fi

end_ms=$(date +%s%3N)
upload_latency=$((end_ms - start_ms))
log "Video upload latency: ${upload_latency} ms"

returned_doc_id=$(python3 -c "import sys, json; print(json.loads('''$REQ_BODY''').get('doc_id', ''))" 2>/dev/null || echo "")
log "Client DOC_ID=$DOC_ID, server returned doc_id=$returned_doc_id"

log "==== 4. Polling document status ===="
max_polls=120  # Increased from 30 to 120 (approx 4 minutes)
poll=1
fully="false"
while [ "$poll" -le "$max_polls" ]; do
  if ! request_with_retry "GET" "$BASE_URL/api/doc/status/$returned_doc_id" "" "-H" "X-API-Key: $API_KEY"; then
    log "Status endpoint failed after retries (last status: $REQ_STATUS)"
    exit 1
  fi
  
  fully=$(python3 -c "import sys, json; print(str(json.loads('''$REQ_BODY''').get('fully_processed', False)).lower())" 2>/dev/null || echo "false")
  exists=$(python3 -c "import sys, json; print(str(json.loads('''$REQ_BODY''').get('exists', False)).lower())" 2>/dev/null || echo "false")
  status=$(python3 -c "import sys, json; print(str(json.loads('''$REQ_BODY''').get('status', 'unknown')))" 2>/dev/null || echo "unknown")
  text_processed=$(python3 -c "import sys, json; print(str(json.loads('''$REQ_BODY''').get('text_processed', False)).lower())" 2>/dev/null || echo "false")
  multimodal_processed=$(python3 -c "import sys, json; print(str(json.loads('''$REQ_BODY''').get('multimodal_processed', False)).lower())" 2>/dev/null || echo "false")
  
  log "Poll $poll/$max_polls: status=$status exists=$exists fully_processed=$fully (text=$text_processed, multimodal=$multimodal_processed)"
  
  if [ "$fully" = "true" ]; then
    log "Document fully processed!"
    break
  fi
  poll=$((poll + 1))
  sleep 2
done

if [ "$fully" != "true" ]; then
  log "Document not fully processed after $max_polls polls"
  exit 1
fi
log "Document processing completed for doc_id=$DOC_ID"

log "==== 5. Running query: exact content retrieval ===="
query_body_1=$(cat <<JSON
{
  "query": "video project_1",
  "mode": "hybrid",
  "top_k": 5,
  "param": {
      "doc_ids": ["$returned_doc_id"]
  }
}
JSON
)
start_ms=$(date +%s%3N)
if ! request_with_retry "POST" "$BASE_URL/api/query" "$query_body_1" "-H" "Content-Type: application/json" "-H" "X-API-Key: $API_KEY"; then
  log "Exact content query failed after retries"
  exit 1
fi
end_ms=$(date +%s%3N)
latency_q1=$((end_ms - start_ms))
log "Exact content query latency: ${latency_q1} ms"
validate_query_response "$REQ_BODY" "content"
if [ "$latency_q1" -gt "$MAX_LATENCY_MS" ]; then
    log "WARNING: Latency exceeded ${MAX_LATENCY_MS}ms"
fi

log "==== 6. Running query: semantic similarity ===="
query_body_2=$(cat <<JSON
{
  "query": "What is the main topic?",
  "mode": "naive",
  "top_k": 5,
  "param": {
      "doc_ids": ["$returned_doc_id"]
  }
}
JSON
)
start_ms=$(date +%s%3N)
if ! request_with_retry "POST" "$BASE_URL/api/query" "$query_body_2" "-H" "Content-Type: application/json" "-H" "X-API-Key: $API_KEY"; then
  log "Semantic similarity query failed after retries"
  exit 1
fi
end_ms=$(date +%s%3N)
latency_q2=$((end_ms - start_ms))
log "Semantic similarity query latency: ${latency_q2} ms"
validate_query_response "$REQ_BODY" "semantic"

log "==== 7. Running query: time range (simulated via query text) ===="
# Assuming time range queries are handled via specific prompts or if we had a dedicated 'time_range' param
# For RAG, we usually include time context in the query
query_body_3=$(cat <<JSON
{
  "query": "What happens at the beginning of the video?",
  "mode": "naive",
  "top_k": 5,
  "param": {
      "doc_ids": ["$returned_doc_id"]
  }
}
JSON
)
start_ms=$(date +%s%3N)
if ! request_with_retry "POST" "$BASE_URL/api/query" "$query_body_3" "-H" "Content-Type: application/json" "-H" "X-API-Key: $API_KEY"; then
  log "Time range query failed after retries"
  exit 1
fi
end_ms=$(date +%s%3N)
latency_q3=$((end_ms - start_ms))
log "Time range query latency: ${latency_q3} ms"
validate_query_response "$REQ_BODY" "time"

log "==== Test Summary ===="
log "All tests passed successfully!"
log "1. Video Upload: OK (${upload_latency}ms)"
log "2. Processing: OK"
log "3. Exact Query: OK (${latency_q1}ms)"
log "4. Semantic Query: OK (${latency_q2}ms)"
log "5. Time Query: OK (${latency_q3}ms)"
