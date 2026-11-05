// Read-only 32-bit register reader for the Debug tab's pin view (needs root for /dev/mem).
//   pins_read 0x2210000+off ... -> one "0xADDR 0xVALUE" line per argument
// /dev/mem is opened O_RDONLY and mapped PROT_READ: this program cannot change any register. Registers are read
// with single aligned 32-bit loads (device memory on arm64 faults on byte or vector access).
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <unistd.h>

int main(int argc, char** argv) {
  int fd = open("/dev/mem", O_RDONLY | O_SYNC);
  if (fd < 0) {
    perror("/dev/mem");
    return 1;
  }
  const long page = sysconf(_SC_PAGESIZE);
  for (int i = 1; i < argc; ++i) {
    const uint64_t addr = strtoull(argv[i], NULL, 0);
    if (addr & 3) {
      printf("0x%llx error unaligned\n", (unsigned long long)addr);
      continue;
    }
    const uint64_t base = addr & ~(uint64_t)(page - 1);
    void* m = mmap(NULL, page, PROT_READ, MAP_SHARED, fd, (off_t)base);
    if (m == MAP_FAILED) {
      printf("0x%llx error mmap\n", (unsigned long long)addr);
      continue;
    }
    const uint32_t v = *(volatile const uint32_t*)((const char*)m + (addr - base));
    printf("0x%llx 0x%08x\n", (unsigned long long)addr, v);
    munmap(m, page);
  }
  close(fd);
  return 0;
}
