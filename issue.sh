# 替换为目标仓库的拥有者和仓库名称，例如：octocat/Spoon-Knife
TARGET_REPO="microsoft/MoGe"

# 获取所有 Issue（包括打开和关闭的），输出为 JSON 格式
# --state all: 获取所有状态的 Issue
# --limit unlimited: 确保获取所有 Issue，没有数量限制
# --json : 指定要提取的字段。你可以根据需要添加或删除字段。
#          常用的字段包括：number, title, body, state, labels, author, createdAt, updatedAt, closedAt, url, comments, assignees, milestones
gh issue list \
  --repo "$TARGET_REPO" \
  --state all \
  --limit 100 \
  --json number,title,body,state,labels,author,createdAt,updatedAt,closedAt,url,comments,assignees,milestone > "${TARGET_REPO//\//_}_issues.json"

echo "所有 Issue 已保存到 ${TARGET_REPO//\//_}_issues.json"