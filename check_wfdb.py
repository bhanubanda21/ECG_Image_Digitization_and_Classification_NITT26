import wfdb

record = wfdb.rdrecord("data/dar_hea/02000/02000_hr")
print("Sampling Frequency:", record.fs)
print("Signal Shape:", record.p_signal.shape)
