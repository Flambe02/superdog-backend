import os
import csv
import re
from math import ceil

def split_content(content, max_length=2000):
    """
    Divise un contenu long en plusieurs parties
    """
    # Si le contenu est déjà assez court, le retourner tel quel
    if len(content.encode('utf-8')) <= max_length:
        return [content]
    
    # Diviser le texte en phrases
    sentences = re.split('([.!?]+)', content)
    parts = []
    current_part = ""
    
    for i in range(0, len(sentences)-1, 2):
        sentence = sentences[i] + (sentences[i+1] if i+1 < len(sentences) else '')
        if len((current_part + sentence).encode('utf-8')) <= max_length:
            current_part += sentence
        else:
            if current_part:
                parts.append(current_part.strip())
            current_part = sentence
    
    if current_part:
        parts.append(current_part.strip())
    
    return parts

def extract_metadata(content):
    metadata = {}
    file_match = re.search(r'CONTENU DE: (.*?)\.txt', content)
    metadata['file_name'] = file_match.group(1) if file_match else "Unknown"
    title_match = re.search(r'Titre: (.*?)(?:\n|$)', content)
    metadata['title'] = title_match.group(1) if title_match else ""
    url_match = re.search(r'URL: (.*?)(?:\n|$)', content)
    metadata['url'] = url_match.group(1) if url_match else ""
    id_match = re.search(r'ID: (.*?)(?:\n|$)', content)
    metadata['id'] = id_match.group(1) if id_match else ""
    return metadata

def process_txt_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='latin-1') as file:
            content = file.read()
    
    metadata = extract_metadata(content)
    is_transcript = 'TRANSCRIPTION:' in content
    entries = []
    
    if is_transcript:
        transcript_match = re.search(r'TRANSCRIPTION:(.*?)(?=={2,}|$)', content, re.DOTALL)
        transcript = transcript_match.group(1).strip() if transcript_match else ""
        
        # Diviser la transcription en parties si nécessaire
        content_parts = split_content(transcript)
        for i, part in enumerate(content_parts, 1):
            entries.append({
                'type': 'transcript',
                'file_name': metadata['file_name'],
                'title': metadata['title'],
                'url': metadata['url'],
                'video_id': metadata['id'],
                'content': part,
                'chapter': f'Part {i}/{len(content_parts)}' if len(content_parts) > 1 else ''
            })
    else:
        current_chapter = ""
        lines = content.split('\n')
        content_buffer = []
        
        for i, line in enumerate(lines):
            if line.strip().startswith(('Lesson', 'Chapter')):
                # Traiter le contenu précédent s'il existe
                if content_buffer:
                    content_text = ' '.join(content_buffer)
                    content_parts = split_content(content_text)
                    for j, part in enumerate(content_parts, 1):
                        entries.append({
                            'type': 'document',
                            'file_name': metadata['file_name'],
                            'chapter': f'{current_chapter} Part {j}/{len(content_parts)}' if len(content_parts) > 1 else current_chapter,
                            'content': part,
                            'title': '',
                            'url': '',
                            'video_id': ''
                        })
                
                current_chapter = line.strip()
                content_buffer = []
            else:
                if line.strip():
                    content_buffer.append(line.strip())
        
        # Traiter le dernier contenu
        if content_buffer:
            content_text = ' '.join(content_buffer)
            content_parts = split_content(content_text)
            for j, part in enumerate(content_parts, 1):
                entries.append({
                    'type': 'document',
                    'file_name': metadata['file_name'],
                    'chapter': f'{current_chapter} Part {j}/{len(content_parts)}' if len(content_parts) > 1 else current_chapter,
                    'content': part,
                    'title': '',
                    'url': '',
                    'video_id': ''
                })
    
    return entries

def write_csv_file(entries, fieldnames, output_file, max_size=4*1024*1024):
    """
    Écrit les entrées dans un fichier CSV en respectant la limite de taille
    """
    current_size = 0
    current_entries = []
    header_size = len(','.join(fieldnames).encode('utf-8')) + 2
    current_size = header_size
    
    for entry in entries:
        entry_size = sum(len(str(value).encode('utf-8')) + 1 for value in entry.values())
        if current_size + entry_size > max_size:
            return current_entries, entries[len(current_entries):]
        
        current_entries.append(entry)
        current_size += entry_size
    
    return current_entries, entries[len(current_entries):]

def convert_to_csv(input_folder, output_base_name):
    all_entries = []
    
    for filename in os.listdir(input_folder):
        if filename.endswith('.txt'):
            file_path = os.path.join(input_folder, filename)
            print(f"Traitement du fichier : {filename}")
            try:
                entries = process_txt_file(file_path)
                all_entries.extend(entries)
            except Exception as e:
                print(f"Erreur lors du traitement de {filename}: {str(e)}")
    
    if all_entries:
        fieldnames = ['type', 'file_name', 'title', 'url', 'video_id', 'chapter', 'content']
        remaining_entries = all_entries
        file_index = 1
        
        while remaining_entries:
            output_file = f"{output_base_name}_{file_index}.csv"
            batch, remaining_entries = write_csv_file(remaining_entries, fieldnames, output_file)
            
            with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(batch)
            
            file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
            print(f"Fichier créé : {output_file} ({file_size_mb:.2f} Mo)")
            file_index += 1
        
        print(f"\nConversion terminée! {file_index-1} fichiers CSV créés")
        print(f"Nombre total d'entrées : {len(all_entries)}")
    else:
        print("Aucune donnée n'a été extraite des fichiers.")

if __name__ == "__main__":
    current_dir = os.getcwd()
    output_base = "output"
    
    print(f"Dossier de travail : {current_dir}")
    print("Démarrage de la conversion...")
    
    convert_to_csv(current_dir, output_base)
