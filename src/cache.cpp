
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the caching functionality of compiled binaries and programs.
//
// =================================================================================================

#include "cache.hpp"

#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <array>

#include "database/database.hpp"
#include "utilities/backend.hpp"

namespace clblast {
// =================================================================================================

template <typename Key, typename Value>
template <typename U>
Value Cache<Key, Value>::Get(const U& key, bool* in_cache) const {
	std::lock_guard<std::mutex> lock(cache_mutex_);
	bool found = false;
	std::array<bool, 3> highest = {false, false, false};
	for (const auto& pair : cache_)
	{
		std::array<bool, 3> attempts = {false, false, false};
		if (std::get<0>(pair.first) == std::get<0>(key))
			attempts[0] = true;
		if (std::get<1>(pair.first) == std::get<1>(key))
			attempts[1] = true;
		if (std::get<2>(pair.first) == std::get<2>(key))
			attempts[2] = true;
		
		size_t highestCount = 0;
		size_t thisCount = 0;
		for (size_t i = 0; i < 3; i += 1)
		{
			if (highest[i]) highestCount += 1;
			if (attempts[i]) thisCount += 1;
		}
		if (thisCount > highestCount)
			highest = attempts;
	}
#if 0
	if (highest[0])
		std::cout << "	device uuids equal\n";
	else
		std::cout << "	device uuids NOT equal\n";
	if (highest[1])
		std::cout << "	precision equal\n";
	else
		std::cout << "	precision NOT equal\n";
	if (highest[2])
		std::cout << "	kernel name? equal\n";
	else
		std::cout << "	kernel name? equal\n";
#endif
#if 1 //__cplusplus >= 201402L
	// generalized std::map::find() of C++14
	auto it = cache_.find(key);
#else
	// O(n) lookup in a vector
	auto it =
			std::find_if(cache_.begin(), cache_.end(), [&](const std::pair<Key, Value>& pair) { return pair.first == key; });
#endif
	if (it == cache_.end()) {
		if (in_cache) {
			*in_cache = false;
		}
		return Value();
	}

	if (in_cache) {
		*in_cache = true;
	}
	return it->second;
}

template <typename Key, typename Value>
void Cache<Key, Value>::Store(Key&& key, Value&& value) {
	std::lock_guard<std::mutex> lock(cache_mutex_);

#if __cplusplus >= 201402L
	// emplace() into a map
	auto r = cache_.emplace(std::move(key), std::move(value));
	if (!r.second) {
		// The object is already in cache. This can happen if two threads both
		// checked the cache for an object, both found that it isn't there, then
		// both produced the object (e.g. a compiled binary) and try to store it
		// in the cache. The first one will succeed normally, the second one will
		// hit this point. We simply return in this case.
		return;
	}
#else
	// emplace_back() into a vector
	cache_.emplace_back(std::move(key), std::move(value));
#endif
}

template <typename Key, typename Value>
void Cache<Key, Value>::Remove(const Key& key) {
	std::lock_guard<std::mutex> lock(cache_mutex_);
#if __cplusplus >= 201402L
	cache_.erase(key);
#else
	auto it = cache_.begin();
	while (it != cache_.end()) {
		if ((*it).first == key) {
			it = cache_.erase(it);
		} else
			++it;
	}
#endif
}

template <typename Key, typename Value>
template <int I1, int I2>
void Cache<Key, Value>::RemoveBySubset(const Key& key) {
	std::lock_guard<std::mutex> lock(cache_mutex_);
	auto it = cache_.begin();
	while (it != cache_.end()) {
		const auto current_key = (*it).first;
		if ((std::get<I1>(key) == std::get<I1>(current_key)) && (std::get<I2>(key) == std::get<I2>(current_key))) {
			it = cache_.erase(it);
		} else
			++it;
	}
}

template <typename Key, typename Value>
void Cache<Key, Value>::Invalidate() {
	std::lock_guard<std::mutex> lock(cache_mutex_);

	cache_.clear();
}

template <typename Key, typename Value>
Cache<Key, Value>& Cache<Key, Value>::Instance() {
	return instance_;
}

template <typename Key, typename Value>
Cache<Key, Value> Cache<Key, Value>::instance_;

// =================================================================================================

template class Cache<BinaryKey, std::string>;
template std::string BinaryCache::Get(const BinaryKeyRef&, bool*) const;

// =================================================================================================

template class Cache<ProgramKey, std::shared_ptr<Program>>;
template std::shared_ptr<Program> ProgramCache::Get(const ProgramKeyRef&, bool*) const;
template void ProgramCache::RemoveBySubset<1, 2>(const ProgramKey&);	// precision and routine name

// =================================================================================================

template class Cache<DatabaseKey, Database>;
template Database DatabaseCache::Get(const DatabaseKeyRef&, bool*) const;

// =================================================================================================
}	// namespace clblast
