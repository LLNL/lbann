// MUST include this
#include <catch2/catch.hpp>

// File being tested
#include <lbann/utils/hash.hpp>

#include <unordered_set>

TEST_CASE ("Testing convenience functions for hashing", "[hash][utilities]") {

  SECTION ("hash_combine") {
    std::unordered_set<size_t> hashes;
    for (size_t seed=0; seed<10; ++seed) {
      hashes.insert(seed);
    }
    for (size_t seed=0; seed<=16; seed+=2) {
      for (int val=-49; val<=49; val+=7) {
        const auto hash = lbann::hash_combine(seed, val);
        CHECK_FALSE(hashes.count(hash));
        hashes.insert(hash);
      }
    }
  }

  SECTION ("enum_hash") {
    enum class Humor { PHLEGMATIC, CHOLERIC, SANGUINE, MELANCHOLIC };
    std::vector<Humor> enum_list = { Humor::MELANCHOLIC, Humor::SANGUINE,
                                     Humor::CHOLERIC, Humor::PHLEGMATIC };
    std::unordered_set<size_t> hashes;
    for (size_t i=0; i<enum_list.size(); ++i) {
      const auto hash = lbann::enum_hash<Humor>()(enum_list[i]);
      CHECK_FALSE(hashes.count(hash));
      hashes.insert(hash);
    }
  }

  SECTION ("pair_hash") {
    std::unordered_set<size_t> hashes;
    for (char i=-12; i<=12; i+=3) {
      for (unsigned long j=0; j<=11209; j+=1019) {
        std::pair<char,unsigned long> val(i,j);
        const auto hash = lbann::pair_hash<char,unsigned long>()(val);
        CHECK_FALSE(hashes.count(hash));
        hashes.insert(hash);
      }
    }
  }

}
