import sqlite3
import json
import os
from datetime import datetime

def inspect_cache_to_md(db_path="verifai_cache.db", output_file="cache_report.md"):
    if not os.path.exists(db_path):
        print(f"❌ Cache database not found at: {db_path}")
        return

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        if not tables:
            print("📭 Cache is empty (no tables found).")
            return

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# VerifAI Cache Report\n\n")
            
            for table in tables:
                table_name = table[0]
                f.write(f"## Table: `{table_name}`\n\n")
                
                cursor.execute(f"SELECT * FROM {table_name} ORDER BY timestamp DESC")
                rows = cursor.fetchall()
                
                # Get column names
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = [col[1] for col in cursor.fetchall()]
                
                if not rows:
                    f.write("*Table is empty*\n\n")
                    continue
                
                # Write table header
                f.write("| Timestamp | Type/Query | Claim Snippet | Verdict / Action | Confidence |\n")
                f.write("|-----------|------------|---------------|------------------|------------|\n")

                for row in rows:
                    data = dict(zip(columns, row))
                    
                    dt = ""
                    if 'timestamp' in data:
                        dt = datetime.fromtimestamp(data['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
                    
                    query_type = data.get('query', 'N/A').replace('|', '&#124;').replace('\n', ' ')
                    claim_snip = data.get('claim', 'N/A')[:60].replace('|', '&#124;').replace('\n', ' ') + "..."
                    
                    verdict_action = "N/A"
                    confidence = "-"
                    
                    if 'result_json' in data:
                        try:
                            res = json.loads(data['result_json'])
                            if 'verdict' in res:
                                verdict_action = f"✅ {res['verdict']}"
                                confidence = f"{res.get('confidence', 0):.2f}"
                            elif 'correction_text' in res:
                                verdict_action = "📝 CORRECTION"
                            else:
                                verdict_action = "📦 Raw JSON/XAI"
                        except:
                            verdict_action = "⚠️ Parse Error"
                    
                    f.write(f"| {dt} | {query_type} | `{claim_snip}` | {verdict_action} | {confidence} |\n")
                
                f.write("\n")

        conn.close()
        print(f"✅ Cache report generated successfully at: {output_file}")
    except Exception as e:
        print(f"❌ Error inspecting cache: {e}")

if __name__ == "__main__":
    inspect_cache_to_md()
