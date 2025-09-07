import music21

s = music21.converter.parse("ninettes-musette.mid")
print(s[1][0][4])
