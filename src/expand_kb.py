import json
import os
import wikipediaapi

def process_sciq():
    """Returns already processed SciQ facts."""
    facts = []
    sciq_path = "data/processed/sciq_kb_facts.jsonl"
    if os.path.exists(sciq_path):
        with open(sciq_path, 'r') as f:
            for line in f:
                fact = json.loads(line)
                fact['source'] = 'SciQ'
                facts.append(fact)
    return facts

def fetch_wikipedia_educational():
    """Fetches summaries from Wikipedia for educational categories."""
    print("Fetching Wikipedia educational summaries...")
    wiki = wikipediaapi.Wikipedia('EducationalFactBot/1.0 (contact@example.com)', 'en')
    
    categories = [
        "Category:Mathematics",
        "Category:Physics",
        "Category:Biology",
        "Category:History",
        "Category:Chemistry",
        "Category:Computer science",
        "Category:Environmental education"
    ]
    
    facts = []
    processed_pages = set()
    
    for cat_name in categories:
        cat = wiki.page(cat_name)
        # Process first 30 members of each category for performance/storage balance
        members = list(cat.categorymembers.values())[:30]
        
        for member in members:
            if member.ns == wikipediaapi.Namespace.MAIN and member.title not in processed_pages:
                try:
                    # Use summary as a verifiable fact
                    text = member.summary.split('\n')[0] # First paragraph/sentence mostly
                    if len(text) > 50: # Sanity check for length
                        facts.append({
                            "fact_id": f"wiki_{len(facts)}",
                            "text": text[:1000], # Limit length but give enough context
                            "source": f"Wikipedia ({member.title})"
                        })
                        processed_pages.add(member.title)
                except Exception as e:
                    print(f"Error processing page {member.title}: {e}")
                    
    return facts

def main():
    os.makedirs("data/processed", exist_ok=True)
    
    all_facts = []
    
    # 1. SciQ
    print("Loading SciQ facts...")
    all_facts.extend(process_sciq())
    
    # 2. Wikipedia
    try:
        all_facts.extend(fetch_wikipedia_educational())
    except Exception as e:
        print(f"Error fetching Wikipedia: {e}")
        
    # Save combined KB
    output_path = "data/processed/combined_edu_kb.jsonl"
    with open(output_path, 'w', encoding='utf-8') as f:
        for fact in all_facts:
            f.write(json.dumps(fact) + '\n')
            
    print(f"Combined KB saved with {len(all_facts)} facts to {output_path}")

if __name__ == "__main__":
    main()
