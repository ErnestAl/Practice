import sys
import re
    

def list_to_XML(lines):
    result = "<speak>\n"
    for line in lines:   
        if line not in ["\n", ""]:
            new_line = re.sub(r'([,;:-])', r'\1<break time="300ms" />', line)
            result += f"<p>\n   {new_line}\n</p>\n"

    result += "\n</speak>"

    return result
    

if len(sys.argv) > 1:
    input_file = sys.argv[1]
else:
    input_file = "raw_text.txt"

if len(sys.argv) > 2:
    output_file = sys.argv[2]
else:
    output_file = "texts/text_00020.txt"

with open(input_file, "r") as f:
    lines = [line.strip() for line in f if line != "\n"]


result = list_to_XML(lines)

with open(output_file, "w", encoding="cp1251") as f:
    f.write(result)