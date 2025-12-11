#!/bin/bash
set -e

BASE_URL="http://127.0.0.1:9622"
API_KEY="raganything-key"
DOC_ID="e2e_test_doc_$(date +%s)"

echo "Waiting for server to be ready..."
sleep 5

echo "1. Testing Health Endpoint..."
HEALTH_STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/health")
if [ "$HEALTH_STATUS" -ne 200 ]; then
  echo "❌ Health check failed with status $HEALTH_STATUS"
  exit 1
fi
echo "✅ Health check passed"

echo "2. Testing Auth..."
AUTH_STATUS=$(curl -s -o /dev/null -w "%{http_code}" -H "X-API-Key: $API_KEY" "$BASE_URL/api/secure")
if [ "$AUTH_STATUS" -ne 200 ]; then
  echo "❌ Auth check failed with status $AUTH_STATUS"
  exit 1
fi
echo "✅ Auth check passed"

echo "3. Testing Content Insertion..."
INSERT_STATUS=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$BASE_URL/api/doc/insert" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d "{
  \"content_list\": [
    {\"type\": \"text\", \"text\": \"RAG-Anything is a powerful tool.\"},
    {\"type\": \"table\", \"table_body\": \"Year,Users\n2024,1000\", \"table_caption\": [\"User growth\"]}
  ],
  \"doc_id\": \"$DOC_ID\",
  \"file_path\": \"test_doc.json\"
}")
if [ "$INSERT_STATUS" -ne 200 ]; then
  echo "❌ Content insertion failed with status $INSERT_STATUS"
  exit 1
fi
echo "✅ Content insertion passed"

echo "4. Testing Document Status..."
# Allow time for processing to start
sleep 2
STATUS_RESPONSE=$(curl -s -w "\n%{http_code}" "$BASE_URL/api/doc/status/$DOC_ID" -H "X-API-Key: $API_KEY")
STATUS_CODE=$(echo "$STATUS_RESPONSE" | tail -n1)
STATUS_BODY=$(echo "$STATUS_RESPONSE" | sed '$d')

if [ "$STATUS_CODE" -ne 200 ]; then
  echo "❌ Status check failed with status $STATUS_CODE"
  exit 1
fi
if ! echo "$STATUS_BODY" | grep -q '"exists":true'; then
    echo "❌ Status check failed: doc_id $DOC_ID not found or status is incorrect."
    echo "Response: $STATUS_BODY"
    exit 1
fi
echo "✅ Status check passed: $STATUS_BODY"


echo "5. Testing Text Query..."
QUERY_RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "$BASE_URL/api/query" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{
  "query": "What is RAG-Anything?",
  "mode": "bypass"
}')
QUERY_CODE=$(echo "$QUERY_RESPONSE" | tail -n1)
QUERY_BODY=$(echo "$QUERY_RESPONSE" | sed '$d')

if [ "$QUERY_CODE" -ne 200 ]; then
    echo "❌ Text query failed with status $QUERY_CODE"
    exit 1
fi
echo "✅ Text query passed"

echo "6. Testing Multimodal Query..."
MULTIMODAL_RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "$BASE_URL/api/query/multimodal" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{
  "query": "Analyze this table",
  "multimodal_content": [
    {
      "type": "table",
      "table_body": "A,B\n1,2",
      "table_caption": ["Test Table"]
    }
  ],
  "mode": "bypass"
}')
MULTIMODAL_CODE=$(echo "$MULTIMODAL_RESPONSE" | tail -n1)
MULTIMODAL_BODY=$(echo "$MULTIMODAL_RESPONSE" | sed '$d')

if [ "$MULTIMODAL_CODE" -ne 200 ]; then
    echo "❌ Multimodal query failed with status $MULTIMODAL_CODE"
    exit 1
fi
echo "✅ Multimodal query passed"

echo "🎉 All tests passed!"
