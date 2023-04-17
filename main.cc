#include <chrono>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>

#include "spdlog/async.h"
#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/spdlog.h"

void sync_logger(int id)
{
	std::string filename = "sync_log_" + std::to_string(id) + ".txt";
	auto logger = spdlog::basic_logger_mt("sync_" + std::to_string(id), filename);
	for (int i = 0; i < 1000; i++)
	{
		logger->info("Message from sync logger {}: {}", id, i);
	}
}

void async_logger(int id)
{
	std::string filename = "async_log_" + std::to_string(id) + ".txt";
	auto async_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(filename);
	auto async_logger = std::make_shared<spdlog::async_logger>("async_" + std::to_string(id), async_sink, spdlog::thread_pool(), spdlog::async_overflow_policy::block);
	for (int i = 0; i < 1000; i++)
	{
		async_logger->info("Message from async logger {}: {}", id, i);
	}
}

int main()
{
	std::vector<std::thread> threads;
	auto start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < 10; i++)
	{
		threads.emplace_back(sync_logger, i);
	}
	for (auto& thread : threads)
	{
		thread.join();
	}
	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> sync_time = end - start;
	std::cout << "Synchronous logging time: " << sync_time.count() << " s\n";

	threads.clear();
	start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < 10; i++)
	{
		threads.emplace_back(async_logger, i);
	}
	for (auto& thread : threads)
	{
		thread.join();
	}
	end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> async_time = end - start;
	std::cout << "Asynchronous logging time: " << async_time.count() << " s\n";

	return 0;
}
