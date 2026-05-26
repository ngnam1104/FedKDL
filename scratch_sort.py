import re

def process():
    with open(r'.docs\related_work.md', 'r', encoding='utf-8') as f:
        content = f.read()

    # Find all table rows
    table_rows = re.findall(r'\|(?! :---).*?\|.*?\|.*?\|.*?\|.*?\|', content)
    headers = ['Tên bài báo', 'Tác giả', 'Vấn đề mục tiêu', 'Phương pháp cốt lõi', 'Nhiệm vụ 1']
    valid_rows = []
    for r in table_rows:
        if not any(h in r for h in headers) and r.strip().startswith('|'):
            valid_rows.append(r.strip())

    unique_rows = {}
    for r in valid_rows:
        col1 = r.split('|')[1].strip()
        # Better key: remove all non-alphanumeric, take up to 60 chars
        key = re.sub(r'[^a-zA-Z0-9]', '', col1)[:60].lower()
        if key not in unique_rows:
            unique_rows[key] = {'row': r, 'col1': col1}

    # Deduplicate details
    lines = content.split('\n')
    details = []
    current = []
    
    # Regex to catch the start of a detail section
    # Matches: **[...], #### [...], **Author...
    start_pattern = r'^\s*(\*\*\[|#### \[|\*\*Anonymous|\*\*Yiyuan|\*\*Yang|\*\*Junchao|\*\*Lin Li|\*\*Laiqiao|\*\*Xinlu)'
    
    for line in lines:
        if re.match(start_pattern, line):
            if current:
                details.append('\n'.join(current).strip())
            current = [line]
        elif current:
            # Do NOT append if it's a table row or a header like "### Nhiệm vụ 1"
            stripped = line.strip()
            if stripped.startswith('|') or stripped.startswith('### Nhiệm vụ') or stripped == '---' or stripped.startswith('Dưới đây là kết quả phân tích'):
                pass
            else:
                current.append(line)

    if current:
        details.append('\n'.join(current).strip())

    unique_details = {}
    for d in details:
        lines_d = d.split('\n')
        if not lines_d: continue
        first_line = lines_d[0]
        # Match inside bracket if present, otherwise just the line
        match = re.search(r'\[(.*?)\]', first_line)
        if match:
            title_text = match.group(1)
        else:
            title_text = first_line.replace('**', '').replace('#### ', '')
            
        key = re.sub(r'[^a-zA-Z0-9]', '', title_text)[:60].lower()
        if key not in unique_details:
            unique_details[key] = {'text': d, 'first_line': title_text}

    print(f"Total unique tables: {len(unique_rows)}")
    print(f"Total unique details: {len(unique_details)}")

    # Extract years for sorting
    def extract_year(text):
        match = re.search(r'\b(19\d\d|20\d\d)\b', text)
        if match:
            return int(match.group(1))
        return 9999

    for k, v in unique_rows.items():
        v['year'] = extract_year(v['col1'])
        
    for k, v in unique_details.items():
        v['year'] = extract_year(v['first_line'])

    sorted_tables = sorted(list(unique_rows.values()), key=lambda x: x['year'])
    sorted_details = sorted(list(unique_details.values()), key=lambda x: x['year'])

    # Write
    out = []
    out.append('### Bảng tóm tắt tổng quan\n')
    out.append('| Tác giả, Năm & Tên bài báo | Vấn đề mục tiêu | Phương pháp cốt lõi | Phân loại | Đóng góp/Kết quả chính |')
    out.append('| :--- | :--- | :--- | :--- | :--- |')
    for t in sorted_tables:
        out.append(t['row'])
    
    out.append('\n---\n')
    out.append('### Phân tích chuyên sâu phương pháp\n')
    for d in sorted_details:
        out.append(d['text'] + '\n')

    with open(r'.docs\related_work.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(out))
        
    print("Done writing .docs\related_work.md")

if __name__ == '__main__':
    process()
