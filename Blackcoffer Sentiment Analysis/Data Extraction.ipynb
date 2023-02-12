{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "cab93c7b",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd \n",
    "import requests, time\n",
    "from bs4 import BeautifulSoup\n",
    "from selenium import webdriver"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "dd70e293",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>URL_ID</th>\n",
       "      <th>URL</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>37.0</td>\n",
       "      <td>https://insights.blackcoffer.com/ai-in-healthc...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>38.0</td>\n",
       "      <td>https://insights.blackcoffer.com/what-if-the-c...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>39.0</td>\n",
       "      <td>https://insights.blackcoffer.com/what-jobs-wil...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>40.0</td>\n",
       "      <td>https://insights.blackcoffer.com/will-machine-...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>41.0</td>\n",
       "      <td>https://insights.blackcoffer.com/will-ai-repla...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>109</th>\n",
       "      <td>146.0</td>\n",
       "      <td>https://insights.blackcoffer.com/blockchain-fo...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>110</th>\n",
       "      <td>147.0</td>\n",
       "      <td>https://insights.blackcoffer.com/the-future-of...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>111</th>\n",
       "      <td>148.0</td>\n",
       "      <td>https://insights.blackcoffer.com/big-data-anal...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>112</th>\n",
       "      <td>149.0</td>\n",
       "      <td>https://insights.blackcoffer.com/business-anal...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>113</th>\n",
       "      <td>150.0</td>\n",
       "      <td>https://insights.blackcoffer.com/challenges-an...</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>114 rows × 2 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     URL_ID                                                URL\n",
       "0      37.0  https://insights.blackcoffer.com/ai-in-healthc...\n",
       "1      38.0  https://insights.blackcoffer.com/what-if-the-c...\n",
       "2      39.0  https://insights.blackcoffer.com/what-jobs-wil...\n",
       "3      40.0  https://insights.blackcoffer.com/will-machine-...\n",
       "4      41.0  https://insights.blackcoffer.com/will-ai-repla...\n",
       "..      ...                                                ...\n",
       "109   146.0  https://insights.blackcoffer.com/blockchain-fo...\n",
       "110   147.0  https://insights.blackcoffer.com/the-future-of...\n",
       "111   148.0  https://insights.blackcoffer.com/big-data-anal...\n",
       "112   149.0  https://insights.blackcoffer.com/business-anal...\n",
       "113   150.0  https://insights.blackcoffer.com/challenges-an...\n",
       "\n",
       "[114 rows x 2 columns]"
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df = pd.read_excel('Input.xlsx')\n",
    "df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "222d65be",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\chhot\\AppData\\Local\\Temp\\ipykernel_8284\\4219914332.py:5: DeprecationWarning: executable_path has been deprecated, please pass in a Service object\n",
      "  driver = webdriver.Chrome('chromedriver.exe')\n"
     ]
    }
   ],
   "source": [
    "x = 37\n",
    "for x in df['URL_ID']:\n",
    "    url = df['URL'][x-37]\n",
    "\n",
    "    driver = webdriver.Chrome('chromedriver.exe')\n",
    "    driver.get(url)\n",
    "    time.sleep(5)\n",
    "    html = driver.page_source\n",
    "\n",
    "    soup = BeautifulSoup(html, 'html.parser')\n",
    "    \n",
    "    for title in soup.find_all('title'):\n",
    "        title = title.get_text()\n",
    "    \n",
    "    lines = []\n",
    "    for data in soup.find_all('p'):\n",
    "        lines.append(data.get_text())\n",
    "    \n",
    "    with open (r\"C:\\Users\\chhot\\Dropbox\\BlackCoffer\\text files\\{}.txt\".format(int(x)),'w',encoding='utf-8') as f:\n",
    "        f.write(title)\n",
    "        f.write('\\n\\n\\n')\n",
    "        for line in lines:\n",
    "            f.write(line)\n",
    "            f.write('\\n\\n')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "24c13c22",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a8cf84b0",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
