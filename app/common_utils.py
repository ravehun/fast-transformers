from datetime import timedelta, datetime


class CommonUtils():
    @staticmethod
    def date_to_idx(cur_date, start="1950-01-01"):
        date_format = "%Y-%m-%d"
        a = datetime.strptime(cur_date, date_format)
        b = datetime.strptime(start, date_format)
        return (a - b).days

    @staticmethod
    def idx_to_date(days, start="1950-01-01"):
        date_format = "%Y-%m-%d"
        cur = datetime.strptime(start, date_format) + timedelta(days=days)
        return cur.isoformat()