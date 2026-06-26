import pexpect
import sys

child = pexpect.spawn('sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y bpftrace bpfcc-tools linux-headers-generic', encoding='utf-8')
child.logfile = sys.stdout

try:
    child.expect('password for thor:', timeout=5)
    child.sendline('1234qwer')
except pexpect.TIMEOUT:
    print("No password prompt")

child.expect(pexpect.EOF, timeout=120)
