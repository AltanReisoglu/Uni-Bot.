import praw
import json
import time
import google.generativeai as genai
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# 1. API Konfigürasyonları
reddit = praw.Reddit(
    client_id="JXTA1GBD-JBJsH30leRpVQ",
    client_secret="H8lr2K3TIAxQr8mp2gTMFDmTi2BrjA",
    user_agent="university-tr-app by u/kucukmonarch"
)

genai.configure(api_key="AIzaSyBymPtSh5RXrPVn4UfXcDlHV_M7nW8X3yA")
model = genai.GenerativeModel('gemini-1.0-pro')
from google import genai

client = genai.Client(api_key="AIzaSyBymPtSh5RXrPVn4UfXcDlHV_M7nW8X3yA")

# 2. Gelişmiş veri çekme
def scrape_reddit_data(subreddits=["UniversityTR", "AskAcademia"], limit_per_sub=50, include_comments=True):
    all_posts = []
    
    for sub_name in subreddits:
        try:
            subreddit = reddit.subreddit(sub_name)
            print(f"Scraping r/{sub_name}...")
            
            for post in subreddit.hot(limit=limit_per_sub):
                if not post.selftext or len(post.selftext) < 50:
                    continue
                
                post_data = {
                    "title": post.title,
                    "content": post.selftext.strip(),
                    "source": f"reddit/r/{sub_name}",
                    "created_utc": post.created_utc,
                    "comments": []
                }
                
                if include_comments:
                    post.comments.replace_more(limit=2)
                    for comment in post.comments.list():
                        if len(comment.body) > 30:
                            post_data["comments"].append(comment.body.strip())
                
                all_posts.append(post_data)
                time.sleep(0.5)  # Rate limiting
            
        except Exception as e:
            print(f"Error scraping r/{sub_name}: {str(e)}")
            continue
    
    return all_posts

# 3. Dinamik soru oluşturma
def generate_custom_prompt(content):
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash", contents=f"""Bu içerik için üniversite öğrencilerine yönelik spesifik bir tavsiye sorusu oluştur:
            
            İçerik: {content[:1500]}
            
            - Soru maksimum 15 kelime olsun
            - Doğal ve samimi bir dil kullan
            - Akademik/kampüs yaşamına odaklan
            - Cevap içerikte mevcut olmalı"""
        )
        return response.text.strip()
    except Exception as e:
        print(f"Prompt generation error: {str(e)}")
        return "Üniversite yaşamıyla ilgili tavsiye verir misin?"

# 4. Paralel işleme
def process_posts_parallel(posts):
    processed = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        
        for post in posts:
            # Ana post için
            combined = f"{post['title']}\n\n{post['content']}"
            futures.append(executor.submit(
                lambda p: {
                    "content": p,
                    "prompt": generate_custom_prompt(p),
                    "source": post['source'],
                    "date": datetime.utcfromtimestamp(post['created_utc']).strftime('%Y-%m-%d')
                },
                combined
            ))
            
            # Yorumlar için
            for comment in post['comments']:
                futures.append(executor.submit(
                    lambda c: {
                        "content": c,
                        "prompt": generate_custom_prompt(c),
                        "source": post['source'] + "_comment",
                        "date": datetime.utcfromtimestamp(post['created_utc']).strftime('%Y-%m-%d')
                    },
                    comment
                ))
        
        for i, future in enumerate(as_completed(futures)):
            try:
                processed.append(future.result())
                if (i+1) % 10 == 0:
                    print(f"Processed {i+1}/{len(futures)} items")
            except Exception as e:
                print(f"Processing error: {str(e)}")
    
    return processed

# 5. JSONL formatına dönüştürme
def create_gemini_jsonl(data):
    formatted = []
    system_instruction = {
        "role": "system",
        "parts": [{
            "text": "Sen bir üniversite danışmanısın. Öğrencilere samimi, pratik ve detaylı akademik/kampüs tavsiyeleri veriyorsun."
        }]
    }
    
    for item in data:
        formatted.append({
            "systemInstruction": system_instruction,
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": item["prompt"]}]
                },
                {
                    "role": "model",
                    "parts": [{"text": item["content"]}]
                }
            ],
            "metadata": {
                "source": item["source"],
                "date": item["date"]
            }
        })
    
    return formatted

# 6. Ana iş akışı
if __name__ == "__main__":
    print("Reddit verileri çekiliyor...")
    posts = scrape_reddit_data(
        subreddits=["UniversityTR", "AskAcademia", "GradSchool","ytu"],
        limit_per_sub=30,
        include_comments=True
    )
    print(f"Toplam {len(posts)} post ve yorumları çekildi.")
    
    print("Dinamik promptlar oluşturuluyor...")
    processed_data = process_posts_parallel(posts)
    
    print("Gemini formatına dönüştürülüyor...")
    gemini_data = create_gemini_jsonl(processed_data)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"reddit_academic_advice_{timestamp}.jsonl"
    
    with open(filename, "w", encoding="utf-8") as f:
        for item in gemini_data:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")
    
    print(f"İşlem tamamlandı! Çıktı: {filename}")
    print(f"Toplam {len(gemini_data)} konuşma kaydedildi.")