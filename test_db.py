# test_db.py
from db_manager import db

print("=== 开始测试用户系统 ===")

# 1. 正常注册一个普通用户
print("\n[1] 注册普通用户")
print(db.register("zhangsan", "13800138000", "2025211990", "123456"))

# 2. 测试字段冲突
print("\n[2] 测试字段冲突")
print(db.register("zhangsan", "13900000000", "2025000000", "123456")) # 用户名冲突
print(db.register("lisi", "13800138000", "2025000000", "123456")) # 手机号冲突

# 3. 注册一个管理员 (role='admin')
print("\n[3] 注册管理员")
print(db.register("admin", "13911112222", "10000001", "admin123", role="admin"))

# 4. 测试三种方式登录
print("\n[4] 测试多方式登录")
print("用用户名登录:", db.login("zhangsan", "123456"))
print("用手机号登录:", db.login("13800138000", "123456"))
print("用学号登录:", db.login("2025211990", "123456"))
print("错误密码登录:", db.login("zhangsan", "wrong"))

# 5. 测试管理员身份校验
print("\n[5] 校验管理员身份")
success, user_info = db.login("admin", "admin123")
if success and user_info['role'] == 'admin':
    print("=> 身份校验通过：欢迎您，管理员！")
else:
    print("=> 身份校验失败")