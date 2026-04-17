from autoplay_v2.cli import build_parser


def test_calibrate_command_defaults_to_interactive():
    parser = build_parser()
    args = parser.parse_args(["calibrate"])
    assert args.left is None
    assert args.top is None
    assert args.width is None
    assert args.height is None


def test_hotkey_command_has_dual_combos():
    parser = build_parser()
    args = parser.parse_args(["hotkey"])
    assert args.play_hotkey == "shift+a+s"
    assert args.calibrate_hotkey == "ctrl+shift+a+s"
