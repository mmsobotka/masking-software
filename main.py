from Application import Application
import gc

if __name__ == "__main__":
    application = Application()
    """---"""
    application.load_interface()
    gc.collect()
