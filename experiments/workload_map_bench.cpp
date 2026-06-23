// Workload-map characterizer for AEGIS-Q.
// Records actual CPU/GPU implementations and explicitly labels missing GPU
// implementations as unsupported; it never fabricates a zero-cost result.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <openssl/evp.h>
#include <openssl/rand.h>
extern "C" {
#include <oqs/oqs.h>
#include "../cuda_aead.h"
}

using Clock = std::chrono::steady_clock;
static constexpr size_t kBlock = 4096;

struct Stats { double p50, p99, throughput; };
static Stats summarize(std::vector<double> us, size_t work) {
  std::sort(us.begin(), us.end());
  size_t n = us.size();
  double p50 = us[n / 2];
  double p99 = us[std::min(n - 1, (size_t)((n - 1) * .99))];
  double mean = 0; for (double v : us) mean += v; mean /= n;
  return {p50, p99, (double)work * 1e6 / mean};
}
static void row(const char *op, const char *target, size_t batch, size_t bytes,
                const char *status, const Stats *s) {
  if (!s) { std::printf("%s,%s,%zu,%zu,%s,,,,\n", op,target,batch,bytes,status); return; }
  std::printf("%s,%s,%zu,%zu,%s,%.3f,%.3f,%.3f\n",op,target,batch,bytes,status,s->p50,s->p99,s->throughput);
}
static void measured(const char *op, const char *target, size_t batch, size_t bytes, std::vector<double> samples, size_t work) {
  Stats s = summarize(std::move(samples), work); row(op,target,batch,bytes,"measured",&s);
}

static bool aes_cpu_once(const uint8_t *in, uint8_t *out, size_t bytes) {
  EVP_CIPHER_CTX *ctx = EVP_CIPHER_CTX_new(); if (!ctx) return false;
  uint8_t key[32] = {}, nonce[12] = {}, aad[28] = {}, tag[16]; int n = 0;
  bool ok = true;
  for (size_t off=0; off<bytes; off += kBlock) {
    size_t len = std::min(kBlock, bytes-off); nonce[11] = (uint8_t)(off/kBlock);
    ok &= EVP_EncryptInit_ex(ctx, EVP_aes_256_gcm(), nullptr, key, nonce) == 1;
    ok &= EVP_EncryptUpdate(ctx, nullptr, &n, aad, sizeof(aad)) == 1;
    ok &= EVP_EncryptUpdate(ctx, out+off, &n, in+off, (int)len) == 1;
    ok &= EVP_EncryptFinal_ex(ctx, out+off+n, &n) == 1;
    ok &= EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_GET_TAG, sizeof(tag), tag) == 1;
    if (!ok) break;
  }
  EVP_CIPHER_CTX_free(ctx); return ok;
}
static bool aes_gpu_once(const uint8_t *in, uint8_t *out, size_t bytes) {
  size_t count=(bytes+kBlock-1)/kBlock; std::vector<uint8_t> nonces(count*12), aads(count*28), tags(count*16);
  std::vector<size_t> offs(count), lens(count); uint8_t key[32]={};
  for(size_t i=0;i<count;i++){ offs[i]=i*kBlock; lens[i]=std::min(kBlock,bytes-offs[i]); nonces[i*12+11]=(uint8_t)i; }
  return skim_cuda_aes256_gcm_batch(key,nonces.data(),aads.data(),in,out,offs.data(),lens.data(),tags.data(),count)==0;
}
static void bench_aes(size_t bytes) {
  std::vector<uint8_t> in(bytes), out(bytes); RAND_bytes(in.data(), (int)bytes);
  std::vector<double> cpu, gpu;
  for(int r=0;r<9;r++){ auto t=Clock::now(); if(!aes_cpu_once(in.data(),out.data(),bytes)) std::abort(); auto e=Clock::now(); if(r) cpu.push_back(std::chrono::duration<double,std::micro>(e-t).count()); }
  measured("aes_gcm", "cpu", (bytes+kBlock-1)/kBlock, bytes, std::move(cpu), bytes);
  if (!skim_cuda_aead_available()) { row("aes_gcm","gpu",(bytes+kBlock-1)/kBlock,bytes,"unsupported",nullptr); return; }
  for(int r=0;r<9;r++){ auto t=Clock::now(); if(!aes_gpu_once(in.data(),out.data(),bytes)) std::abort(); auto e=Clock::now(); if(r) gpu.push_back(std::chrono::duration<double,std::micro>(e-t).count()); }
  measured("aes_gcm", "gpu", (bytes+kBlock-1)/kBlock, bytes, std::move(gpu), bytes);
}

static void bench_kem(size_t batch) {
  OQS_KEM *kem=OQS_KEM_new(OQS_KEM_alg_ml_kem_768); if(!kem){ row("ml_kem_keygen","cpu",batch,0,"unsupported",nullptr); return; }
  std::vector<uint8_t> pk(kem->length_public_key), sk(kem->length_secret_key), ct(kem->length_ciphertext*batch), ss(kem->length_shared_secret*batch);
  std::vector<double> kg, en, de;
  for(int r=0;r<7;r++){ auto t=Clock::now(); for(size_t i=0;i<batch;i++) OQS_KEM_keypair(kem,pk.data(),sk.data()); auto e=Clock::now(); if(r)kg.push_back(std::chrono::duration<double,std::micro>(e-t).count()); }
  OQS_KEM_keypair(kem,pk.data(),sk.data());
  for(int r=0;r<7;r++){ auto t=Clock::now(); for(size_t i=0;i<batch;i++) OQS_KEM_encaps(kem,ct.data()+i*kem->length_ciphertext,ss.data()+i*kem->length_shared_secret,pk.data()); auto e=Clock::now(); if(r)en.push_back(std::chrono::duration<double,std::micro>(e-t).count()); }
  for(int r=0;r<7;r++){ auto t=Clock::now(); for(size_t i=0;i<batch;i++) OQS_KEM_decaps(kem,ss.data()+i*kem->length_shared_secret,ct.data()+i*kem->length_ciphertext,sk.data()); auto e=Clock::now(); if(r)de.push_back(std::chrono::duration<double,std::micro>(e-t).count()); }
  measured("ml_kem_keygen","cpu",batch,0,std::move(kg),batch); measured("ml_kem_encaps","cpu",batch,0,std::move(en),batch); measured("ml_kem_decaps","cpu",batch,0,std::move(de),batch);
  row("ml_kem_keygen","gpu",batch,0,"unsupported",nullptr); row("ml_kem_encaps","gpu",batch,0,"unsupported",nullptr); row("ml_kem_decaps","gpu",batch,0,"unsupported",nullptr);
  OQS_MEM_cleanse(sk.data(),sk.size()); OQS_KEM_free(kem);
}

static void bench_sig(size_t batch) {
  OQS_SIG *sig=OQS_SIG_new(OQS_SIG_alg_ml_dsa_65); if(!sig){ row("ml_dsa_sign","cpu",batch,0,"unsupported",nullptr); return; }
  std::vector<uint8_t> pk(sig->length_public_key),sk(sig->length_secret_key),msg(32,7),s(sig->length_signature*batch); size_t slen=sig->length_signature;
  OQS_SIG_keypair(sig,pk.data(),sk.data()); std::vector<double> sign,verify;
  for(int r=0;r<7;r++){auto t=Clock::now();for(size_t i=0;i<batch;i++){size_t n=0;OQS_SIG_sign(sig,s.data()+i*sig->length_signature,&n,msg.data(),msg.size(),sk.data());slen=n;}auto e=Clock::now();if(r)sign.push_back(std::chrono::duration<double,std::micro>(e-t).count());}
  for(int r=0;r<7;r++){auto t=Clock::now();for(size_t i=0;i<batch;i++)OQS_SIG_verify(sig,msg.data(),msg.size(),s.data()+i*sig->length_signature,slen,pk.data());auto e=Clock::now();if(r)verify.push_back(std::chrono::duration<double,std::micro>(e-t).count());}
  measured("ml_dsa_sign","cpu",batch,0,std::move(sign),batch); measured("ml_dsa_verify","cpu",batch,0,std::move(verify),batch);
  row("ml_dsa_sign","gpu",batch,0,"unsupported",nullptr); row("ml_dsa_verify","gpu",batch,0,"unsupported",nullptr);
  OQS_MEM_cleanse(sk.data(),sk.size()); OQS_SIG_free(sig);
}

static void bench_sha256(size_t leaves) {
  std::vector<uint8_t> in(leaves*32,1), out(leaves*32); std::vector<double> samples;
  for(int r=0;r<7;r++){auto t=Clock::now();for(size_t i=0;i<leaves;i++){size_t n=0;EVP_Q_digest(nullptr,"SHA256",nullptr,in.data()+i*32,32,out.data()+i*32,&n);}auto e=Clock::now();if(r)samples.push_back(std::chrono::duration<double,std::micro>(e-t).count());}
  measured("sha256_leaf","cpu",leaves,leaves*32,std::move(samples),leaves);
  row("sha256_leaf","gpu",leaves,leaves*32,"unsupported",nullptr);
}

int main(){
  std::puts("operation,target,batch,bytes,status,p50_us,p99_us,throughput_per_s");
  for(size_t b: {size_t(4096),size_t(16384),size_t(65536),size_t(262144),size_t(1048576),size_t(4194304),size_t(16777216)}) bench_aes(b);
  for(size_t n: {size_t(1),size_t(16),size_t(64),size_t(256),size_t(1024),size_t(4096)}) { bench_kem(n); bench_sig(n); }
  for(size_t n: {size_t(1),size_t(16),size_t(256),size_t(4096),size_t(65536),size_t(1048576)}) bench_sha256(n);
  skim_cuda_executor_shutdown();
}
