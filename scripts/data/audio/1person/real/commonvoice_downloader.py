import requests
import json
import shutil
import progressbar
from p_tqdm import p_map
import numpy as np
import time


# cv_languages_json = {
#     "ab": "Abkhaz",
#     "af": "Afrikaans",
#     "sq": "Albanian",
#     "am": "Amharic",
#     "ar": "Arabic",
#     "hy-AM": "Armenian",
#     "as": "Assamese",
#     "ast": "Asturian",
#     "az": "Azerbaijani",
#     "bas": "Basaa",
#     "ba": "Bashkir",
#     "eu": "Basque",
#     "be": "Belarusian",
#     "bn": "Bengali",
#     "br": "Breton",
#     "bg": "Bulgarian",
#     "yue": "Cantonese",
#     "ca": "Catalan",
#     "ckb": "Central Kurdish",
#     "zh-CN": "Chinese (China)",
#     "zh-HK": "Chinese (Hong Kong)",
#     "zh-TW": "Chinese (Taiwan)",
#     "cv": "Chuvash",
#     "cs": "Czech",
#     "da": "Danish",
#     "dv": "Dhivehi",
#     "dyu": "Dioula",
#     "nl": "Dutch",
#     "en": "English",
#     "myv": "Erzya",
#     "eo": "Esperanto",
#     "et": "Estonian",
#     "fi": "Finnish",
#     "fr": "French",
#     "fy-NL": "Frisian",
#     "gl": "Galician",
#     "ka": "Georgian",
#     "de": "German",
#     "el": "Greek",
#     "gn": "Guarani",
#     "cnh": "Hakha Chin",
#     "ha": "Hausa",
#     "he": "Hebrew",
#     "mrj": "Hill Mari",
#     "hi": "Hindi",
#     "hu": "Hungarian",
#     "is": "Icelandic",
#     "ig": "Igbo",
#     "id": "Indonesian",
#     "ia": "Interlingua",
#     "ga-IE": "Irish",
#     "it": "Italian",
#     "ja": "Japanese",
#     "kab": "Kabyle",
#     "kk": "Kazakh",
#     "rw": "Kinyarwanda",
#     "ko": "Korean",
#     "kmr": "Kurmanji Kurdish",
#     "ky": "Kyrgyz",
#     "lo": "Lao",
#     "lv": "Latvian",
#     "lt": "Lithuanian",
#     "lg": "Luganda",
#     "mk": "Macedonian",
#     "ml": "Malayalam",
#     "mt": "Maltese",
#     "mr": "Marathi",
#     "mhr": "Meadow Mari",
#     "mdf": "Moksha",
#     "mn": "Mongolian",
#     "ne-NP": "Nepali",
#     "nn-NO": "Norwegian Nynorsk",
#     "oc": "Occitan",
#     "or": "Odia",
#     "ps": "Pashto",
#     "fa": "Persian",
#     "pl": "Polish",
#     "pt": "Portuguese",
#     "pa-IN": "Punjabi",
#     "quy": "Quechua Chanka",
#     "ro": "Romanian",
#     "rm-sursilv": "Romansh Sursilvan",
#     "rm-vallader": "Romansh Vallader",
#     "ru": "Russian",
#     "sah": "Sakha",
#     "sat": "Santali (Ol Chiki)",
#     "skr": "Saraiki",
#     "sc": "Sardinian",
#     "sr": "Serbian",
#     "sk": "Slovak",
#     "sl": "Slovenian",
#     "hsb": "Sorbian, Upper",
#     "es": "Spanish",
#     "sw": "Swahili",
#     "sv-SE": "Swedish",
#     "nan-tw": "Taiwanese (Minnan)",
#     "zgh": "Tamazight",
#     "ta": "Tamil",
#     "tt": "Tatar",
#     "th": "Thai",
#     "tig": "Tigre",
#     "ti": "Tigrinya",
#     "tok": "Toki Pona",
#     "tr": "Turkish",
#     "tk": "Turkmen",
#     "tw": "Twi",
#     "uk": "Ukrainian",
#     "ur": "Urdu",
#     "ug": "Uyghur",
#     "uz": "Uzbek",
#     "vi": "Vietnamese",
#     "vot": "Votic",
#     "cy": "Welsh",
#     "yo": "Yoruba"
#   }


cv_languages_json = {
    "en": "English",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "es": "Spanish"
}


def download_file(url, local_filename, language_name):
    n_chunk = 1
    with requests.get(url, stream=True) as r:
        block_size = 1024*1024*10
        file_size = int(r.headers.get('Content-Length', None))
        num_bars = np.ceil(file_size / (n_chunk * block_size))
        bar =  progressbar.ProgressBar(widgets=[language_name, ' ', progressbar.Percentage(), ' ', progressbar.Bar()], maxval=num_bars).start()
        with open(local_filename, 'wb') as f:
            for i, chunk in enumerate(r.iter_content(chunk_size=n_chunk * block_size)):
                f.write(chunk)
                bar.update(i+1)
                # Add a little sleep so you can see the bar progress
                time.sleep(0.01)

    return local_filename


def download_language_data(language_key):
    # print(language_key)
    language_name = cv_languages_json[language_key]
    metadata_template = "https://commonvoice.mozilla.org/api/v1/bucket/dataset/cv-corpus-15.0-2023-09-08%2Fcv-corpus-15.0-2023-09-08-{}.tar.gz" 
    metadata_url = metadata_template.format(language_key)
    r = requests.get(metadata_url)
    metadata = json.loads(r.content)
    actual_file_url = metadata["url"]
    # print("metadata :",language_name, " ", actual_file_url)
    local_path = "/projects/whissle/datasets/cv/{}.tar.gz".format(language_key)

    download_file(actual_file_url, local_path, language_name)


all_language_keys = list(cv_languages_json.keys())
p_map(download_language_data, all_language_keys)

# for _ln_key in cv_languages_json.keys():
#     download_language_data(_ln_key)
#     # break

