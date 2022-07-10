# import requests, bs4

# i = 1
# authors = set()
# while True:
#     try:
#             result = requests.get(("http://quotes.toscrape.com/page/{}/").format(i))

#             soup = bs4.BeautifulSoup(result.text, "lxml")

#             for k in soup.select(".author"):
#                 authors.add(k.text)
                
#             for j in authors:
#                 print(j)
                
#             i = i + 1
        
#     except:
#         print(result.text)
#         print("asdasf")
#         break
    
# from PIL import Image

# img_word_matrix = Image.open("Python/word_matrix.png")
# img_mask        = Image.open("Python/mask.png")

# img_mask =  img_mask.resize((1015,559))

# img_word_matrix.putalpha(50)

# img_mask.paste(im = img_word_matrix, box = (0,0),mask = img_word_matrix)

# img_mask.show()

# import csv

# a_file = open("Python/find_the_link.csv", encoding="utf-8")

# a_file_reader = csv.reader(a_file)

# a_file_data = list(a_file_reader)

# i = 0
# link = ""
# for t in a_file_data:
#     link = link + t[i]
#     i = i + 1

# print(link)

# import PyPDF2

# pdf_file = open("Python/Find_the_Phone_Number.pdf", "rb")

# pdf_file_reader = PyPDF2.PdfFileReader(pdf_file)

# print(pdf_file_reader.getPage(0).extract_text())

# pdf_file_page = pdf_file_reader.getPage(pageNumber)


dict = {x : x**3 for x in range(0,5)}
print(dict)