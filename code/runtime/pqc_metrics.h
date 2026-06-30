#ifndef PQC_METRICS_H
#define PQC_METRICS_H

double pqc_metrics_time_us(void);
int pqc_metrics_open_for_storage(const char *storage_dir);
void pqc_metrics_close(void);
void pqc_log(const char *fmt, ...);

#endif /* PQC_METRICS_H */
