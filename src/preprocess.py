
from src import constants as const


class FillInStrategy:
    """
    Used for filling out the missing data on the weekends and NAs.
    NAs are replaces with the the data from the last weekend.
        Possible BUG: first price is NA.
        Fix: Manually replace NA with some value.
    """

    def __init__(self, first_weekend):
        self.first_weekend = first_weekend

    def fill(self, data, fill_strategy='default'):
        filled_data = []
        last_price = data[0]
        for price in data:
            # Replace NAs with the available last price
            if price == -1:
                filled_data.append(last_price)
            else:
                if data.index(price) % 5 == self.first_weekend:
                    filled_data += getattr(self, '_FillInStrategy__'+fill_strategy+'_fill')(last_price, price)
                else:
                    filled_data.append(price)
                last_price = price
        return filled_data

    def __default_fill(self, last_price, price):
        """
        Do not fill in weekend prices, just skip them.
        """
        return [price]

    def __gate_fill(self, last_price, price):
        """
        Fill weekends with last available price.
        """
        return [last_price, last_price, price]

    def __linear_fill(self, last_price, price):
        """
        Linear interpolation between Monday price and last available price.
        """
        diff = price - last_price
        return [last_price+diff/3, last_price+diff*2/3, price]

    def __average_fill(self, last_price, price):
        """
        Set weekend prices to the average of Monday price and last available price.
        """
        avg = (last_price + price) / 2
        return [avg, avg, price]


class Parser:

    fill_strategy = ''
    strategy = FillInStrategy(1)

    def __init__(self, fill_strategy):
        self.fill_strategy = fill_strategy

    @staticmethod
    def __extract_and_encode_day(date):
        """
        The encoding method works for any data starting after 2000th year
        :param date:
        :return encoded_day:
        """
        encoded_day = 0
        for x in date.split(',')[0].split('-'):
            print(x)
        year, month, day = [int(x) for x in date.split(',')[0].split('-')]
        for i in range(2000, year):
            encoded_day += 366 if i/4 is 0 else 365
        for i in range(0, month-1):
            encoded_day += const.months[i]
        encoded_day += day

        return encoded_day

    def __determine_first_weekend(self, data):
        """
        Determines the index where first missing weekend data should be injected.
        Possible BUG: when NA is somewhere in the first 5 prices.
        Fix: Manually replace NA with some value.
        :param data:
        :return:
        """
        last_day = self.__extract_and_encode_day(data[0])
        for line in data[1:]:
            if self.__extract_and_encode_day(line) != last_day-1:
                self.strategy.first_weekend = data.index(line)
                break

    def parse_file(self, filename):
        """
        Parses input file "filename" and reverses if prices are in
        ascending order (not starting from constants.CURRENT_YEAR)
        :param filename:
        :return feature data as vector:
        """
        data = open(const.data_path+filename, 'r').read().split('\n')
        if int(data[0].split(',')[0].split('-')[0]) != const.CURRENT_YEAR:
            data = list(reversed(data))

        #self.__determine_first_weekend(data)
        # Files have to be aligned to start at the same point in time
        self.strategy.first_weekend = 1
        return self.strategy.fill([float(x.split(',')[1]) if 'N/A' not in x and x != '' else -1 for x in data], self.fill_strategy)
