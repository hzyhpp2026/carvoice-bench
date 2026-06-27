"""Optional manual DashScope import probe, not a pytest dependency."""


def main() -> None:
    import dashscope

    print(dir(dashscope))
    print("Dashscope imported successfully!")


if __name__ == "__main__":
    main()
