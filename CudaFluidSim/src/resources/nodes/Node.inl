#pragma once

#include "Node.hpp"

#include <filesystem>
#include <set>
#include <unordered_set>
#include <map>
#include <unordered_map>
#include <list>

#include "resources/Resource.hpp"
#include "utils/Enumerate.hpp"
#include "utils/String.hpp"
#include "math/Math.hpp"

template<typename T>
T Node::Get() const {
	T value;
	*this >> value;
	return value;
}

template<typename T>
T Node::GetWithFallback(const T& fallback) const {
	if (!IsValid())
		return fallback;

	return Get<T>();
}

template<typename T>
bool Node::Get(T& dest) const {
	if (!IsValid())
		return false;

	*this >> dest;
	return true;
}

template<typename T, typename K>
bool Node::GetWithFallback(T& dest, const K& fallback) const {
	if (!IsValid()) {
		dest = fallback;
		return false;
	}

	*this >> dest;
	return true;
}

template<typename T>
bool Node::Get(T&& dest) const {
	if (!IsValid())
		return false;

	*this >> dest;
	return true;
}

template<typename T, typename K>
bool Node::GetWithFallback(T&& dest, const K& fallback) const {
	if (!IsValid()) {
		dest = fallback;
		return false;
	}

	*this >> dest;
	return true;
}

template<typename T>
void Node::Set(const T& value) {
	*this << value;
}

template<typename T>
void Node::Set(T&& value) {
	*this << value;
}

template<typename T>
Node& Node::Append(const T& value) {
	AddProperty() << value;
	return *this;
}

template<typename ...Args>
Node& Node::Append(const Args &...args) {
	(Append(args), ...);
	return *this;
}

template<typename T>
Node& Node::operator=(const T& rhs) {
	Set(rhs);
	return *this;
}

inline const Node& operator>>(const Node& node, Node& object) {
	object = node;
	return node;
}

inline Node& operator<<(Node& node, const Node& object) {
	node = object;
	return node;
}

/*const Node &operator>>(const Node &node, std::nullptr_t &object) {
	object = nullptr;
	return node;
}*/

inline Node& operator<<(Node& node, const std::nullptr_t& object) {
	node.SetValue("");
	node.SetType(NodeType::Null);
	return node;
}

template<typename T, std::enable_if_t<std::is_pointer_v<T>, int> = 0>
Node& operator<<(Node& node, const T object) {
	if (!object)
		return node << nullptr;

	node << *object;
	return node;
}

template<typename T>
const Node& operator>>(const Node& node, std::unique_ptr<T>& object) {
	object = std::make_unique<T>();
	node >> *object;
	return node;
}

template<typename T>
Node& operator<<(Node& node, const std::unique_ptr<T>& object) {
	if (!object)
		return node << nullptr;

	node << *object;
	return node;
}

template<typename T>
const Node& operator>>(const Node& node, std::shared_ptr<T>& object) {
	// TODO: Abstract Resource streams out from shared_ptr.
	if constexpr (std::is_base_of_v<Resource, T>) {
		object = T::Create(node);
		return node;
	}
	else {
		object = std::make_shared<T>();
		node >> *object;
		return node;
	}
}

template<typename T>
Node& operator<<(Node& node, const std::shared_ptr<T>& object) {
	if (!object)
		return node << nullptr;

	node << *object;
	return node;
}

inline const Node& operator>>(const Node& node, bool& object) {
	object = String::From<bool>(node.GetValue());
	return node;
}

inline Node& operator<<(Node& node, bool object) {
	node.SetValue(String::To(object));
	node.SetType(NodeType::Boolean);
	return node;
}

template<typename T, std::enable_if_t<std::is_arithmetic_v<T> || std::is_enum_v<T>, int> = 0>
const Node & operator>>(const Node & node, T & object) {
	object = String::From<T>(node.GetValue());
	return node;
}

template<typename T, std::enable_if_t<std::is_arithmetic_v<T> || std::is_enum_v<T>, int> = 0>
Node & operator<<(Node & node, T object) {
	node.SetValue(String::To(object));
	node.SetType(std::is_floating_point_v<T> ? NodeType::Decimal : NodeType::Integer);
	return node;
}

inline Node& operator<<(Node& node, std::string_view string) {
	node.SetValue(std::string(string));
	node.SetType(NodeType::String);
	return node;
}

inline const Node& operator>>(const Node& node, std::string& string) {
	string = node.GetValue();
	return node;
}

inline Node& operator<<(Node& node, const std::string& string) {
	node.SetValue(string);
	node.SetType(NodeType::String);
	return node;
}

inline const Node& operator>>(const Node& node, std::wstring& string) {
	string = String::ConvertUtf16(node.GetValue());
	return node;
}

inline Node& operator<<(Node& node, const std::wstring& string) {
	node.SetValue(String::ConvertUtf8(string));
	node.SetType(NodeType::String);
	return node;
}

inline const Node& operator>>(const Node& node, std::filesystem::path& object) {
	object = node.GetValue();
	return node;
}

inline Node& operator<<(Node& node, const std::filesystem::path& object) {
	auto str = object.string();
	std::replace(str.begin(), str.end(), '\\', '/');
	node.SetValue(str);
	node.SetType(NodeType::String);
	return node;
}

template<typename T, typename K>
const Node& operator>>(const Node& node, std::chrono::duration<T, K>& duration) {
	T x;
	node >> x;
	duration = std::chrono::duration<T, K>(x);
	return node;
}

template<typename T, typename K>
Node& operator<<(Node& node, const std::chrono::duration<T, K>& duration) {
	return node << duration.count();
}

template<typename T>
const Node& operator>>(const Node& node, std::chrono::time_point<T>& timePoint) {
	typename std::chrono::time_point<T>::duration x;
	node >> x;
	timePoint = std::chrono::time_point<T>(x);
	return node;
}

template<typename T>
Node& operator<<(Node& node, const std::chrono::time_point<T>& timePoint) {
	return node << timePoint.time_since_epoch();
}

template<typename T, typename K>
const Node& operator>>(const Node& node, std::pair<T, K>& pair) {
	node["first"].Get(pair.first);
	node["second"].Get(pair.second);
	return node;
}

template<typename T, typename K>
Node& operator<<(Node& node, const std::pair<T, K>& pair) {
	node["first"].Set(pair.first);
	node["second"].Set(pair.second);
	return node;
}

template<typename T>
const Node& operator>>(const Node& node, std::optional<T>& optional) {
	if (node.GetType() != NodeType::Null) {
		T x;
		node >> x;
		optional = std::move(x);
	}
	else {
		optional = std::nullopt;
	}

	return node;
}

template<typename T>
Node& operator<<(Node& node, const std::optional<T>& optional) {
	if (optional)
		return node << *optional;

	return node << nullptr;
}

template<typename T>
const Node& operator>>(const Node& node, std::vector<T>& vector) {
	vector.clear();
	vector.reserve(node.GetProperties().size());

	for (const auto& [propertyName, property] : node.GetProperties())
		property >> vector.emplace_back();

	return node;
}

template<typename T>
Node& operator<<(Node& node, const std::vector<T>& vector) {
	for (const auto& x : vector)
		node.AddProperty() << x;

	node.SetType(NodeType::Array);
	return node;
}

template<typename T>
const Node& operator>>(const Node& node, std::set<T>& set) {
	set.clear();
	auto where = set.end();

	for (const auto& [propertyName, property] : node.GetProperties()) {
		T x;
		property >> x;
		where = set.insert(where, std::move(x));
	}

	return node;
}

template<typename T>
Node& operator<<(Node& node, const std::set<T>& set) {
	for (const auto& x : set)
		node.AddProperty() << x;

	node.SetType(NodeType::Array);
	return node;
}

template<typename T>
const Node& operator>>(const Node& node, std::unordered_set<T>& set) {
	set.clear();
	auto where = set.end();

	for (const auto& [propertyName, property] : node.GetProperties()) {
		T x;
		property >> x;
		where = set.insert(where, std::move(x));
	}

	return node;
}

template<typename T>
Node& operator<<(Node& node, const std::unordered_set<T>& set) {
	for (const auto& x : set)
		node.AddProperty() << x;

	node.SetType(NodeType::Array);
	return node;
}

template<typename T>
const Node& operator>>(const Node& node, std::multiset<T>& set) {
	set.clear();
	auto where = set.end();

	for (const auto& [propertyName, property] : node.GetProperties()) {
		T x;
		property >> x;
		where = set.insert(where, std::move(x));
	}

	return node;
}

template<typename T>
Node& operator<<(Node& node, const std::multiset<T>& set) {
	for (const auto& x : set)
		node.AddProperty() << x;

	node.SetType(NodeType::Array);
	return node;
}

template<typename T>
const Node& operator>>(const Node& node, std::unordered_multiset<T>& set) {
	set.clear();
	auto where = set.end();

	for (const auto& [propertyName, property] : node.GetProperties()) {
		T x;
		property >> x;
		where = set.insert(where, std::move(x));
	}

	return node;
}

template<typename T>
Node& operator<<(Node& node, const std::unordered_multiset<T>& set) {
	for (const auto& x : set)
		node.AddProperty() << x;

	node.SetType(NodeType::Array);
	return node;
}

template<typename T, std::size_t Size>
const Node& operator>>(const Node& node, std::array<T, Size>& array) {
	array = {};

	for (auto [i, propertyPair] : Enumerate(node.GetProperties()))
		propertyPair.second >> array[i];

	return node;
}

template<typename T, std::size_t Size>
Node& operator<<(Node& node, const std::array<T, Size>& array) {
	for (const auto& x : array)
		node.AddProperty() << x;

	node.SetType(NodeType::Array);
	return node;
}

template<typename T>
const Node& operator>>(const Node& node, std::list<T>& list) {
	list.clear();

	for (const auto& [propertyName, property] : node.GetProperties()) {
		T x;
		property >> x;
		list.emplace_back(std::move(x));
	}

	return node;
}

template<typename T>
Node& operator<<(Node& node, const std::list<T>& list) {
	for (const auto& x : list)
		node.AddProperty() << x;

	node.SetType(NodeType::Array);
	return node;
}

template<typename T, typename K>
const Node& operator>>(const Node& node, std::map<T, K>& map) {
	map.clear();
	auto where = map.end();

	for (const auto& [propertyName, property] : node.GetProperties()) {
		std::pair<T, K> pair;
		pair.first = String::From<T>(propertyName);
		property >> pair.second;
		where = map.insert(where, std::move(pair));
	}

	return node;
}

template<typename T, typename K>
Node& operator<<(Node& node, const std::map<T, K>& map) {
	for (const auto& pair : map)
		node.AddProperty(String::To(pair.first)) << pair.second;

	node.SetType(NodeType::Object);
	return node;
}

template<typename T, typename K>
const Node& operator>>(const Node& node, std::unordered_map<T, K>& map) {
	map.clear();
	auto where = map.end();

	for (const auto& [propertyName, property] : node.GetProperties()) {
		std::pair<T, K> pair;
		pair.first = String::From<T>(propertyName);
		property >> pair.second;
		where = map.insert(where, std::move(pair));
	}

	return node;
}

template<typename T, typename K>
Node& operator<<(Node& node, const std::unordered_map<T, K>& map) {
	for (const auto& pair : map)
		node.AddProperty(String::To(pair.first)) << pair.second;

	node.SetType(NodeType::Object);
	return node;
}

template<typename T, typename K>
const Node& operator>>(const Node& node, std::multimap<T, K>& map) {
	map.clear();
	auto where = map.end();

	for (const auto& [propertyName, property] : node.GetProperties()) {
		std::pair<T, K> pair;
		pair.first = String::From<T>(propertyName);
		property >> pair.second;
		where = map.insert(where, std::move(pair));
	}

	return node;
}

template<typename T, typename K>
Node& operator<<(Node& node, const std::multimap<T, K>& map) {
	for (const auto& pair : map)
		node.AddProperty(String::To(pair.first)) << pair.second;

	node.SetType(NodeType::Object);
	return node;
}

template<typename T, typename K>
const Node& operator>>(const Node& node, std::unordered_multimap<T, K>& map) {
	map.clear();
	auto where = map.end();

	for (const auto& [propertyName, property] : node.GetProperties()) {
		std::pair<T, K> pair;
		pair.first = String::From<T>(propertyName);
		property >> pair.second;
		where = map.insert(where, std::move(pair));
	}

	return node;
}

template<typename T, typename K>
Node& operator<<(Node& node, const std::unordered_multimap<T, K>& map) {
	for (const auto& pair : map)
		node.AddProperty(String::To(pair.first)) << pair.second;

	node.SetType(NodeType::Object);
	return node;
}

// glm operators
const Node& operator>>(const Node& node, glm::vec3& v);
Node& operator<<(Node& node, const glm::vec3& v);