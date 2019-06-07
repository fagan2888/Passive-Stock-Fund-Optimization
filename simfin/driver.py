# -*- coding: utf-8 -*-
"""
Set up driver to avoid requisite passing of API key directly through user
input. All parameters should be set manually below, and are passed into
machinery which hits the SimFin API

04/28/2019
Jared Berry
"""

KEY = ''
QUARTERS = ['Q1', 'Q2', 'Q3', 'Q4']
YEARS = ['2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019']
STATEMENT_TYPES = ['pl', 'bs', 'cf']
PULL_SQL = False
USE_PROCESS_FILE = True
