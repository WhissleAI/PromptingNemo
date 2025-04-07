from nemo_text_processing.inverse_text_normalization.inverse_normalize import InverseNormalizer
inverse_normalizer = InverseNormalizer(lang='en')
spoken = "seven"
un_normalized = inverse_normalizer.inverse_normalize(spoken, verbose=True)
print(un_normalized)