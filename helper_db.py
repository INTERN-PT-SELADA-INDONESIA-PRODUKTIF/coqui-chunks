import sqlite3

def header_db(PATH_AUDIO):
    # Buat koneksi ke database
    connection = sqlite3.connect('audio_transcript.db')
    cursor = connection.cursor()

    # Buat pernyataan SQL untuk melakukan insert
    insert_query = '''
    INSERT INTO audio_header (PATH_AUDIO)
    VALUES (?);
    '''

    # Eksekusi pernyataan insert
    cursor.execute(insert_query, (PATH_AUDIO,))

    # Simpan perubahan dan tutup koneksi
    connection.commit()
    cursor.close()
    connection.close()

    return print("sukses")


def max_key():
    # Buat koneksi ke database
    connection = sqlite3.connect('audio_transcript.db')
    cursor = connection.cursor()

    # Buat pernyataan SQL untuk mendapatkan id paling besar dari tabel
    max_id_query = '''
    SELECT MAX(id) FROM audio_header;
    '''

    # Eksekusi pernyataan dan ambil hasilnya
    cursor.execute(max_id_query)
    max_id = cursor.fetchone()[0]

    # Tutup koneksi
    connection.close()
    return max_id


def chunk_db(path_audio, transcript, speaker):
    audio_id = max_key() #foreign key
    connection = sqlite3.connect('audio_transcript.db')
    cursor = connection.cursor()

    insert_query = '''
    INSERT INTO chunk_table (path_audio, transcript, speaker, audio_id)
    VALUES(?, ?, ?, ?);
    '''

    # Eksekusi pernyataan insert
    cursor.execute(insert_query, (path_audio, transcript, speaker, audio_id))

    # Simpan perubahan dan tutup koneksi
    connection.commit()
    cursor.close()
    connection.close()

    return print("sukses")