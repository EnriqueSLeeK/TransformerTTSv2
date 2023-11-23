
import os

import tacotron_cleaner as text_cleaner
from g2p_en import G2p

# G2p: used to convert text to phoneme, converted
# in hope to help in the synthesis

# tacotrong_cleaner: Used to clean and normalize
# english_text using the english_cleaners


# split line
def split_line(line, delimiter=','):
    return line.split(delimiter)


# Preprocess text(clean and normalize) then save it on a file
# This function consider a line is composed with this structure
#  ID|Text transcript|Normalized Text
def text_and_preprocess(text_file,
                        delimiter=',',
                        output_preprocess="preprocessed_text.txt",
                        output_dir="preprocessed_data"):

    path = os.path.join(output_dir, output_preprocess)
    g2p = G2p()

    with open(path, "w") as output_file:
        with open(text_file, "r") as text_file:
            for line in text_file:

                splitted = split_line(line, delimiter)
                splitted_line_len = len(splitted)
                if (splitted_line_len <= 1):
                    continue

                # Extract the normalized text and pass it to the
                # tacotron english cleaner
                normalized_text = splitted[-1].replace("-", "")\
                    .replace("..", "")\

                normalized_text = text_cleaner.cleaners.\
                    english_cleaners(normalized_text)

                # Use G2p to convert the text to phoneme
                out_line = g2p(normalized_text)

                output_file.write(f"{splitted[0]}|")
                output_file.write(f"{'#'.join(out_line)}\n")
