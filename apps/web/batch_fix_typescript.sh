#!/bin/bash

echo "批量修复TypeScript错误..."

# 修复空数组返回
find src -name "*.ts" -name "*.tsx" -exec grep -l "return {} as.*\[\]" {} \; | while read file; do
    sed -i '' 's/return {} as \([^[]*\)\[\]/return [] as \1[]/g' "$file"
done

# 修复测试文件中的ApiResponse错误
find src -name "*.test.tsx" -exec grep -l "mockResolvedValue({})" {} \; | while read file; do
    sed -i '' 's/mockResolvedValue({})/mockResolvedValue({ success: true, data: {} })/g' "$file"
done

# 修复Option组件的className问题
find src -name "*.tsx" -exec grep -l "<Option.*className" {} \; | while read file; do
    sed -i '' 's/<Option[^>]*className="[^"]*"[^>]*value="\([^"]*\)"[^>]*>/<Option value="\1">/g' "$file"
done

echo "批量修复完成！"
