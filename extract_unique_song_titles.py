import csv

# Path to the CSV file
csv_path = 'abrsm_lmth25.csv'

unique_titles = set()

with open(csv_path, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        # Extract both title_piece_1 and title_piece_2
        title1 = row.get('title_piece_1', '').strip()
        title2 = row.get('title_piece_2', '').strip()
        if title1:
            unique_titles.add(title1)
        if title2:
            unique_titles.add(title2)


# Save to a text file
output_path = 'unique_song_titles.txt'
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(f"Unique song titles ({len(unique_titles)}):\n")
    for title in sorted(unique_titles):
        f.write(title + '\n')

print(f"Saved {len(unique_titles)} unique song titles to {output_path}")
