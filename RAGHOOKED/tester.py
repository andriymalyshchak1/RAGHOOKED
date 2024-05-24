import csv
import PyPDF2

def load_data_from_pdf(file_path):
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()
    return text

def save_text_to_csv(text, output_file):
    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([text])

file_path = 'University_of_Texas_at_Austin.pdf'
output_file = 'output.csv'

pdf_text = load_data_from_pdf(file_path)
save_text_to_csv(pdf_text, output_file)

