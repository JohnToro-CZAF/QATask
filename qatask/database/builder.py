from .sqlite import SQLiteDatabase
def build_database(cfg):
    if cfg.type == "sqlite":
        return SQLiteDatabase(cfg)
    else:
        assert cfg.type == "sqlite", "NotImplemented database{}".format(cfg.type)
