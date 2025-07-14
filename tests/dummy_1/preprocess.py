def preprocess(source_csv_path, preprocessed_csv_path):
    """
    A dummy preprocessor that does nothing but copy the file.
    """
    import shutil
    shutil.copyfile(source_csv_path, preprocessed_csv_path)
