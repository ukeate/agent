const fs = require('fs');
const path = require('path');

// 读取App.tsx文件
const appTsxPath = path.join(__dirname, 'App.tsx');
const appTsxContent = fs.readFileSync(appTsxPath, 'utf8');

// 提取所有的页面导入
const importRegex = /lazy\(\(\) => import\(['"]\.\/pages\/([^'"]+)['"]\)/g;
const matches = [];
let match;

while ((match = importRegex.exec(appTsxContent)) !== null) {
  matches.push(match[1]);
}

console.log('Found', matches.length, 'page imports in App.tsx');

// 检查每个文件是否存在
const missing = [];
const existing = [];

matches.forEach(pagePath => {
  const fullPath = path.join(__dirname, 'pages', `${pagePath}.tsx`);
  if (fs.existsSync(fullPath)) {
    existing.push(pagePath);
  } else {
    missing.push(pagePath);
  }
});

console.log('\nExisting pages:', existing.length);
console.log('Missing pages:', missing.length);

if (missing.length > 0) {
  console.log('\nMISSING PAGES:');
  missing.forEach(page => {
    console.log('  -', page + '.tsx');
  });
} else {
  console.log('\n✅ All pages exist!');
}