import cloudscraper
from bs4 import BeautifulSoup
import json
import time
import google.generativeai as genai
from concurrent.futures import ThreadPoolExecutor, as_completed

# Güncellenmiş Gemini API ayarları
GEMINI_API_KEY = "AIzaSyBymPtSh5RXrPVn4UfXcDlHV_M7nW8X3yA"  # Buraya kendi API key'inizi girin
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.0-pro')  # Güncellenmiş model adı
from google import genai

client = genai.Client(api_key=GEMINI_API_KEY)


def scrape_eksi_topic(topic_slug, max_pages=5):
    scraper = cloudscraper.create_scraper()
    entries = []
    
    for page in range(1, max_pages + 1):
        try:
            url = f"https://eksisozluk.com/{topic_slug}?p={page}"
            print(f"Scraping page {page}: {url}")
            
            resp = scraper.get(url)
            resp.raise_for_status()  # HTTP hataları için

            soup = BeautifulSoup(resp.content, "html.parser")
            entry_divs = soup.find_all("div", class_="content")

            if not entry_divs:
                print(f"Sayfa {page}'de entry bulunamadı.")
                continue

            for div in entry_divs:
                text = div.get_text(strip=True)
                if 30 < len(text) < 2000:  # Çok kısa ve çok uzun entry'leri filtrele
                    entries.append(text)

            time.sleep(2)  # Çok hızlı istekleri önlemek için

        except Exception as e:
            print(f"Sayfa {page} işlenirken hata oluştu: {str(e)}")
            continue

    return entries

def generate_question_for_entry(entry):
    try:
        prompt = f"""Bu entry'yi oku ve üniversite öğrencilerine yönelik akademik tavsiye isteyen 
        tek bir doğal soru oluştur. Soru entry'nin içeriğiyle doğrudan ilgili olsun ama 
        entry'nin tamamını tekrar etmesin. Max 20 kelime.
        
        Entry: {entry[:1000]}"""
        response = client.models.generate_content(
            model="gemini-2.5-flash", contents=prompt
        )

        
        return response.text.strip().replace('"', '')  # Tırnak işaretlerini temizle
    except Exception as e:
        print(f"Soru oluşturulurken hata: {str(e)}")
        # Basit bir fallback mekanizması
        if "ders" in entry.lower():
            return "Derslerle ilgili tavsiye verir misin?"
        return "Üniversite hayatıyla ilgili genel tavsiye verir misin?"

def process_entries_parallel(entries, max_workers=3):  # Daha düşük worker sayısı
    questions = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(generate_question_for_entry, entry): i for i, entry in enumerate(entries)}
        
        for future in as_completed(futures):
            try:
                question = future.result()
                idx = futures[future]
                questions.append((idx, question))
                print(f"Processed entry {idx+1}/{len(entries)}")
            except Exception as e:
                print(f"Error processing entry: {str(e)}")
    
    # Orijinal sıraya göre yeniden düzenle
    questions.sort(key=lambda x: x[0])
    return [q for _, q in questions]

def format_for_gemini_jsonl(entries, questions):
    system_instruction = {
        "role": "system",
        "parts": [{
            "text": "Sen bir eğitim danışmanısın. Üniversite öğrencilerine samimi, pratik ve detaylı akademik tavsiyeler veriyorsun."
        }]
    }

    data = []
    for entry, question in zip(entries, questions):
        item = {
            "systemInstruction": system_instruction,
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": question}]
                },
                {
                    "role": "model",
                    "parts": [{"text": entry}]
                }
            ]
        }
        data.append(item)
    return data

def save_jsonl(data_list, filename):
    with open(filename, "w", encoding="utf-8") as f:
        for item in data_list:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")

if __name__ == "__main__":
    topic = "universiteye-yeni-baslayacaklara-tavsiyeler--383385"
    max_pages = 15  # Kaç sayfa taranacak

    print("Eksi Sözlük verisi çekiliyor...")
    entries = scrape_eksi_topic(topic, max_pages=max_pages)
    print(f"{len(entries)} entry bulundu.")

    print("Entry'ler için sorular oluşturuluyor (bu biraz zaman alabilir)...")
    questions = process_entries_parallel(entries)
    
    print("Veri Gemini formatına dönüştürülüyor...")
    formatted_data = format_for_gemini_jsonl(entries, questions)

    output_file = "akademik_tavsiyeler_gemini.jsonl"
    save_jsonl(formatted_data, output_file)
    print(f"JSONL dosyası kaydedildi: {output_file}")
    print(f"Toplam {len(formatted_data)} konuşma kaydedildi.")